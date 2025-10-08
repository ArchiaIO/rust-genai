use crate::adapter::adapters::support::get_api_key;
use crate::adapter::bedrock::BedrockStreamer;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{
	Binary, BinarySource, ChatOptionsSet, ChatRequest, ChatResponse, ChatRole, ChatStream, ChatStreamResponse,
	ContentPart, MessageContent, PromptTokensDetails, ReasoningEffort, ToolCall, Usage,
};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::WebResponse;
use crate::{Headers, ModelIden};
use crate::{Result, ServiceTarget};
use reqwest::RequestBuilder;
use serde_json::{Value, json};
use tracing::warn;
use value_ext::JsonValueExt;

pub struct BedrockAdapter;

const REASONING_LOW: u32 = 1024;
const REASONING_MEDIUM: u32 = 8000;
const REASONING_HIGH: u32 = 24000;

// Max tokens based on Bedrock Claude models
const MAX_TOKENS_64K: u32 = 64000;
const MAX_TOKENS_32K: u32 = 32000;
const MAX_TOKENS_8K: u32 = 8192;
const MAX_TOKENS_4K: u32 = 4096;

const MODELS: &[&str] = &[
	"anthropic.claude-3-5-sonnet-20240620-v1:0",
	"anthropic.claude-3-5-sonnet-20241022-v2:0",
	"anthropic.claude-3-5-haiku-20241022-v1:0",
	"anthropic.claude-3-opus-20240229-v1:0",
	"us.anthropic.claude-3-5-sonnet-20241022-v2:0",
	"eu.anthropic.claude-3-5-sonnet-20241022-v2:0",
];

impl BedrockAdapter {
	pub const API_KEY_DEFAULT_ENV_NAME: &str = "AWS_ACCESS_KEY_ID";
}

impl Adapter for BedrockAdapter {
	fn default_endpoint() -> Endpoint {
		// Default to a placeholder - users should use ServiceTargetResolver for actual Bedrock endpoints
		const BASE_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com/";
		Endpoint::from_static(BASE_URL)
	}

	fn default_auth() -> AuthData {
		// For Bedrock, auth is typically handled via AWS credentials
		// Users should use ServiceTargetResolver for custom auth
		AuthData::from_env(Self::API_KEY_DEFAULT_ENV_NAME)
	}

	async fn all_model_names(_kind: AdapterKind) -> Result<Vec<String>> {
		Ok(MODELS.iter().map(|s| s.to_string()).collect())
	}

	fn get_service_url(_model: &ModelIden, _service_type: ServiceType, endpoint: Endpoint) -> Result<String> {
		// For Bedrock, the URL is typically set via ServiceTargetResolver
		// The endpoint might be a secured proxy that handles the AWS-specific routing
		// We just return the base_url as-is since proxies handle the path construction
		Ok(endpoint.base_url().to_string())
	}

	fn to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		chat_req: ChatRequest,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		let ServiceTarget { endpoint, auth, model } = target;

		// Get API key (might be AWS credentials or proxy auth token)
		let api_key = get_api_key(auth, &model)?;

		// Get URL
		let url = Self::get_service_url(&model, service_type, endpoint)?;

		// Headers - can be customized via ChatOptions.extra_headers
		let mut headers = Headers::from(vec![
			("Authorization".to_string(), format!("Bearer {}", api_key)),
			("Content-Type".to_string(), "application/json".to_string()),
		]);

		// Merge extra headers if provided (e.g., X-SecuredHost, X-Op, X-Model-Id)
		if let Some(extra_headers) = options_set.extra_headers() {
			headers.merge_with(extra_headers);
		}

		// Build request payload using Anthropic format (Bedrock uses same format)
		let BedrockRequestParts {
			system,
			messages,
			tools,
		} = Self::into_bedrock_request_parts(chat_req.clone())?;

		// Determine max_tokens
		let max_tokens = options_set
			.max_tokens()
			.unwrap_or_else(|| Self::default_max_tokens(&model.model_name));

		// Build the payload
		let mut payload = json!({
			"model": &*model.model_name,
			"messages": messages,
			"max_tokens": max_tokens,
		});

		// Add system prompt if present
		if let Some(system_content) = system {
			payload["system"] = system_content;
		}

		// Add tools if present
		if let Some(tools_arr) = tools {
			payload["tools"] = tools_arr;
		}

		// Add temperature if specified
		if let Some(temperature) = options_set.temperature() {
			payload["temperature"] = json!(temperature);
		}

		// Add top_p if specified
		if let Some(top_p) = options_set.top_p() {
			payload["top_p"] = json!(top_p);
		}

		// Add stop sequences if specified
		if !options_set.stop_sequences().is_empty() {
			payload["stop_sequences"] = json!(options_set.stop_sequences());
		}

		// Add thinking/reasoning effort if supported
		if let Some(reasoning_effort) = options_set.reasoning_effort() {
			if let Some(budget) = Self::reasoning_effort_to_budget(&reasoning_effort, &model.model_name) {
				payload["thinking"] = json!({
					"type": "enabled",
					"budget_tokens": budget
				});
			}
		}

		// For streaming, add stream parameter
		if matches!(service_type, ServiceType::ChatStream) {
			payload["stream"] = json!(true);
		}

		Ok(WebRequestData { url, headers, payload })
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		let WebResponse { mut body, .. } = web_response;

		// Extract content array
		let content_arr = body.x_take::<Vec<Value>>("/content")?;

		// Parse content parts
		let mut content_parts: Vec<ContentPart> = Vec::new();
		let mut reasoning_content: Option<String> = None;

		for mut item in content_arr {
			match item.x_get_str("type") {
				Ok("text") => {
					if let Ok(text) = item.x_get_str("text") {
						content_parts.push(ContentPart::Text(text.to_string()));
					}
				}
				Ok("thinking") => {
					if let Ok(thinking) = item.x_get_str("thinking") {
						reasoning_content = Some(thinking.to_string());
					}
				}
				Ok("tool_use") => {
					let call_id = item.x_get_str("id")?.to_string();
					let fn_name = item.x_get_str("name")?.to_string();
					let fn_arguments: Value = item.x_take("input")?;

					content_parts.push(ContentPart::ToolCall(ToolCall {
						call_id,
						fn_name,
						fn_arguments,
					}));
				}
				_ => {
					// Unknown content type, skip
				}
			}
		}

		// Parse usage
		let usage = Self::into_usage(&mut body)?;

		// Capture raw body if requested
		let captured_raw_body = options_set.capture_raw_body().unwrap_or_default().then(|| body);

		Ok(ChatResponse {
			content: MessageContent::from_parts(content_parts),
			reasoning_content,
			model_iden: model_iden.clone(),
			provider_model_iden: model_iden,
			usage,
			captured_raw_body,
		})
	}

	fn to_chat_stream(
		model_iden: ModelIden,
		reqwest_builder: RequestBuilder,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatStreamResponse> {
		let bedrock_stream = BedrockStreamer::new(reqwest_builder, model_iden.clone(), options_set);
		let chat_stream = ChatStream::from_inter_stream(bedrock_stream);
		Ok(ChatStreamResponse {
			model_iden,
			stream: chat_stream,
		})
	}

	fn to_embed_request_data(
		_service_target: ServiceTarget,
		_embed_req: crate::embed::EmbedRequest,
		_options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		Err(crate::Error::Internal(
			"Bedrock embedding not yet supported".to_string(),
		))
	}

	fn to_embed_response(
		_model_iden: ModelIden,
		_web_response: WebResponse,
		_options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::embed::EmbedResponse> {
		Err(crate::Error::Internal(
			"Bedrock embedding not yet supported".to_string(),
		))
	}
}

// region:    --- Support

impl BedrockAdapter {
	pub(super) fn into_usage(body: &mut Value) -> Result<Usage> {
		let mut usage = Usage::default();

		if let Ok(input_tokens) = body.x_get::<i32>("/usage/input_tokens") {
			usage.prompt_tokens = Some(input_tokens);

			// Check for cached tokens
			if let Ok(cache_creation) = body.x_get::<i32>("/usage/cache_creation_input_tokens") {
				let mut details = PromptTokensDetails::default();
				details.cached_tokens = Some(cache_creation);
				usage.prompt_tokens_details = Some(details);
			}
		}

		if let Ok(output_tokens) = body.x_get::<i32>("/usage/output_tokens") {
			usage.completion_tokens = Some(output_tokens);
		}

		// Compute total
		if usage.prompt_tokens.is_some() || usage.completion_tokens.is_some() {
			usage.total_tokens = Some(usage.prompt_tokens.unwrap_or(0) + usage.completion_tokens.unwrap_or(0));
		}

		Ok(usage)
	}

	fn into_bedrock_request_parts(chat_req: ChatRequest) -> Result<BedrockRequestParts> {
		let mut system: Option<Value> = None;
		let mut messages: Vec<Value> = Vec::new();

		for msg in chat_req.messages {
			match msg.role {
				ChatRole::System => {
					// Bedrock uses same system format as Anthropic
					let text = msg.content.joined_texts().unwrap_or_default();
					system = Some(json!(text));
				}
				ChatRole::User | ChatRole::Assistant => {
					let mut content_parts: Vec<Value> = Vec::new();

					for part in msg.content.parts() {
						match part {
							ContentPart::Text(text) => {
								content_parts.push(json!({
									"type": "text",
									"text": text
								}));
							}
							ContentPart::Binary(Binary {
								content_type, source, ..
							}) => match source {
								BinarySource::Base64(base64_data) => {
									content_parts.push(json!({
										"type": "image",
										"source": {
											"type": "base64",
											"media_type": content_type,
											"data": base64_data
										}
									}));
								}
								BinarySource::Url(url) => {
									content_parts.push(json!({
										"type": "image",
										"source": {
											"type": "url",
											"url": url
										}
									}));
								}
							},
							ContentPart::ToolCall(tool_call) => {
								content_parts.push(json!({
									"type": "tool_use",
									"id": tool_call.call_id,
									"name": tool_call.fn_name,
									"input": tool_call.fn_arguments
								}));
							}
							ContentPart::ToolResponse(tool_response) => {
								content_parts.push(json!({
									"type": "tool_result",
									"tool_use_id": tool_response.call_id,
									"content": tool_response.content
								}));
							}
						}
					}

					let role_str = match msg.role {
						ChatRole::User => "user",
						ChatRole::Assistant => "assistant",
						_ => "user", // fallback
					};
					messages.push(json!({
						"role": role_str,
						"content": content_parts
					}));
				}
				ChatRole::Tool => {
					warn!("Tool role not directly supported in Bedrock format, converting to user message");
					// Convert tool responses to user messages
					let mut content_parts: Vec<Value> = Vec::new();
					for part in msg.content.parts() {
						if let ContentPart::ToolResponse(tool_response) = part {
							content_parts.push(json!({
								"type": "tool_result",
								"tool_use_id": tool_response.call_id,
								"content": tool_response.content
							}));
						}
					}
					messages.push(json!({
						"role": "user",
						"content": content_parts
					}));
				}
			}
		}

		// Convert tools if present
		let tools = if let Some(ref tools_vec) = chat_req.tools {
			if !tools_vec.is_empty() {
				let tools_arr: Vec<Value> = tools_vec
					.iter()
					.map(|tool| {
						let mut tool_obj = json!({
							"name": &tool.name,
						});
						if let Some(desc) = &tool.description {
							tool_obj["description"] = json!(desc);
						}
						if let Some(schema) = &tool.schema {
							tool_obj["input_schema"] = schema.clone();
						}
						tool_obj
					})
					.collect();
				Some(json!(tools_arr))
			} else {
				None
			}
		} else {
			None
		};

		Ok(BedrockRequestParts {
			system,
			messages,
			tools,
		})
	}

	fn default_max_tokens(model_name: &str) -> u32 {
		let model_lower = model_name.to_lowercase();

		if model_lower.contains("opus-4") || model_lower.contains("claude-4-opus") {
			MAX_TOKENS_32K
		} else if model_lower.contains("3-opus") || model_lower.contains("3-haiku") {
			MAX_TOKENS_4K
		} else if model_lower.contains("3-5-haiku") {
			MAX_TOKENS_8K
		} else {
			// Default for sonnet and newer models
			MAX_TOKENS_64K
		}
	}

	fn reasoning_effort_to_budget(reasoning_effort: &ReasoningEffort, model_name: &str) -> Option<u32> {
		// Only certain models support thinking/reasoning
		let model_lower = model_name.to_lowercase();
		if !model_lower.contains("sonnet") && !model_lower.contains("opus") {
			return None;
		}

		match reasoning_effort {
			ReasoningEffort::Low => Some(REASONING_LOW),
			ReasoningEffort::Medium => Some(REASONING_MEDIUM),
			ReasoningEffort::High => Some(REASONING_HIGH),
			ReasoningEffort::Budget(tokens) => Some(*tokens),
			ReasoningEffort::Minimal => None, // Minimal reasoning not supported for Bedrock
		}
	}
}

struct BedrockRequestParts {
	system: Option<Value>,
	messages: Vec<Value>,
	tools: Option<Value>,
}

// endregion: --- Support

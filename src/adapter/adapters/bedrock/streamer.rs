use crate::adapter::adapters::support::{StreamerCapturedData, StreamerOptions};
use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use crate::chat::{ChatOptionsSet, ToolCall, Usage};
use crate::{Error, ModelIden, Result};
use base64::Engine as Base64Engine;
use bytes::Bytes;
use reqwest::RequestBuilder;
use serde_json::Value;
use std::error::Error as StdError;
use std::pin::Pin;
use std::task::{Context, Poll};
use value_ext::JsonValueExt;

/// BedrockStreamer handles AWS Bedrock event stream format
/// AWS streams are base64-encoded JSON events, different from Anthropic's SSE format
pub struct BedrockStreamer {
	state: StreamerState,
	options: StreamerOptions,
	captured_data: StreamerCapturedData,
	in_progress_block: InProgressBlock,
}

enum StreamerState {
	NotStarted(RequestBuilder),
	Streaming { response: reqwest::Response, done: bool },
	Error,
}

enum InProgressBlock {
	Text,
	ToolUse { id: String, name: String, input: String },
	Thinking,
}

impl BedrockStreamer {
	pub fn new(reqwest_builder: RequestBuilder, model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		Self {
			state: StreamerState::NotStarted(reqwest_builder),
			options: StreamerOptions::new(model_iden, options_set),
			captured_data: Default::default(),
			in_progress_block: InProgressBlock::Text,
		}
	}

	/// Parse AWS event stream chunk (base64-encoded events in JSON)
	fn parse_aws_event(&mut self, chunk: Bytes) -> Result<Vec<InterStreamEvent>> {
		let mut events = Vec::new();
		let chunk_str = String::from_utf8_lossy(chunk.as_ref());

		// AWS event stream format: {"bytes": "base64_encoded_data"}
		// Multiple events can be in a single chunk
		let mut search_start = 0;
		while let Some(start) = chunk_str[search_start..].find("{\"bytes\":\"") {
			let absolute_start = search_start + start;
			if let Some(end) = chunk_str[absolute_start..].find("\"}") {
				let json_str = &chunk_str[absolute_start..absolute_start + end + 2];
				search_start = absolute_start + end + 2;

				// Parse the wrapper JSON
				if let Ok(wrapper) = serde_json::from_str::<Value>(json_str) {
					if let Some(bytes_b64) = wrapper.get("bytes").and_then(|b| b.as_str()) {
						// Decode base64
						if let Ok(decoded) = base64::prelude::BASE64_STANDARD.decode(bytes_b64) {
							if let Ok(decoded_str) = String::from_utf8(decoded) {
								// Parse the actual event payload
								if let Ok(payload) = serde_json::from_str::<Value>(&decoded_str) {
									// Process the event based on type
									if let Some(event_type) = payload.get("type").and_then(|t| t.as_str()) {
										if let Some(event) = self.process_bedrock_event(event_type, payload.clone())? {
											events.push(event);
										}
									}
								}
							}
						}
					}
				}
			} else {
				break;
			}
		}

		Ok(events)
	}

	/// Process a single Bedrock event
	fn process_bedrock_event(&mut self, event_type: &str, payload: Value) -> Result<Option<InterStreamEvent>> {
		match event_type {
			"message_start" => {
				self.capture_usage(&payload)?;
				Ok(Some(InterStreamEvent::Start))
			}
			"message_delta" => {
				self.capture_usage(&payload)?;
				Ok(None) // Continue processing
			}
			"content_block_start" => {
				if let Some(content_block) = payload.get("content_block") {
					match content_block.x_get_str("/type") {
						Ok("text") => self.in_progress_block = InProgressBlock::Text,
						Ok("thinking") => self.in_progress_block = InProgressBlock::Thinking,
						Ok("tool_use") => {
							self.in_progress_block = InProgressBlock::ToolUse {
								id: content_block.x_get_str("/id")?.to_string(),
								name: content_block.x_get_str("/name")?.to_string(),
								input: String::new(),
							};
						}
						_ => {}
					}
				}
				Ok(None)
			}
			"content_block_delta" => {
				if let Some(delta) = payload.get("delta") {
					match &mut self.in_progress_block {
						InProgressBlock::Text => {
							if let Ok(text) = delta.x_get_str("/text") {
								// Capture if requested
								if self.options.capture_content {
									match self.captured_data.content {
										Some(ref mut c) => c.push_str(text),
										None => self.captured_data.content = Some(text.to_string()),
									}
								}
								return Ok(Some(InterStreamEvent::Chunk(text.to_string())));
							}
						}
						InProgressBlock::ToolUse { input, .. } => {
							if let Ok(partial_json) = delta.x_get_str("/partial_json") {
								input.push_str(partial_json);
							}
						}
						InProgressBlock::Thinking => {
							if let Ok(thinking) = delta.x_get_str("/thinking") {
								// Capture if requested
								if self.options.capture_reasoning_content {
									match self.captured_data.reasoning_content {
										Some(ref mut r) => r.push_str(thinking),
										None => self.captured_data.reasoning_content = Some(thinking.to_string()),
									}
								}
								return Ok(Some(InterStreamEvent::ReasoningChunk(thinking.to_string())));
							}
						}
					}
				}
				Ok(None)
			}
			"content_block_stop" => {
				match std::mem::replace(&mut self.in_progress_block, InProgressBlock::Text) {
					InProgressBlock::ToolUse { id, name, input } => {
						let tc = ToolCall {
							call_id: id,
							fn_name: name,
							fn_arguments: serde_json::from_str(&input).unwrap_or(Value::Null),
						};

						// Capture if requested
						if self.options.capture_tool_calls {
							match self.captured_data.tool_calls {
								Some(ref mut t) => t.push(tc.clone()),
								None => self.captured_data.tool_calls = Some(vec![tc.clone()]),
							}
						}

						return Ok(Some(InterStreamEvent::ToolCallChunk(tc)));
					}
					_ => {}
				}
				Ok(None)
			}
			"message_stop" => {
				// Mark as done
				if let StreamerState::Streaming { ref mut done, .. } = self.state {
					*done = true;
				}

				// Capture the usage
				let captured_usage = if self.options.capture_usage {
					self.captured_data.usage.take().map(|mut usage| {
						// Compute the total if any of input/output are not null
						if usage.prompt_tokens.is_some() || usage.completion_tokens.is_some() {
							usage.total_tokens =
								Some(usage.prompt_tokens.unwrap_or(0) + usage.completion_tokens.unwrap_or(0));
						}
						usage
					})
				} else {
					None
				};

				let inter_stream_end = InterStreamEnd {
					captured_usage,
					captured_text_content: self.captured_data.content.take(),
					captured_reasoning_content: self.captured_data.reasoning_content.take(),
					captured_tool_calls: self.captured_data.tool_calls.take(),
				};

				Ok(Some(InterStreamEvent::End(inter_stream_end)))
			}
			"ping" => Ok(None), // Ignore ping events
			_ => {
				tracing::warn!("Unknown Bedrock event type: {}", event_type);
				Ok(None)
			}
		}
	}

	fn capture_usage(&mut self, payload: &Value) -> Result<()> {
		if self.options.capture_usage {
			// Try message_start format first
			if let Ok(input_tokens) = payload.x_get::<i32>("/message/usage/input_tokens") {
				let val = self
					.captured_data
					.usage
					.get_or_insert(Usage::default())
					.prompt_tokens
					.get_or_insert(0);
				*val += input_tokens;
			} else if let Ok(input_tokens) = payload.x_get::<i32>("/usage/input_tokens") {
				// Try message_delta format
				let val = self
					.captured_data
					.usage
					.get_or_insert(Usage::default())
					.prompt_tokens
					.get_or_insert(0);
				*val += input_tokens;
			}

			// Same for output tokens
			if let Ok(output_tokens) = payload.x_get::<i32>("/message/usage/output_tokens") {
				let val = self
					.captured_data
					.usage
					.get_or_insert(Usage::default())
					.completion_tokens
					.get_or_insert(0);
				*val += output_tokens;
			} else if let Ok(output_tokens) = payload.x_get::<i32>("/usage/output_tokens") {
				let val = self
					.captured_data
					.usage
					.get_or_insert(Usage::default())
					.completion_tokens
					.get_or_insert(0);
				*val += output_tokens;
			}
		}

		Ok(())
	}
}

impl futures::Stream for BedrockStreamer {
	type Item = Result<InterStreamEvent>;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		// Clone model_iden early to avoid borrow issues
		let model_iden = self.options.model_iden.clone();

		// First, initialize if needed
		if let StreamerState::NotStarted(_) = self.state {
			let builder = match std::mem::replace(&mut self.state, StreamerState::Error) {
				StreamerState::NotStarted(b) => b,
				_ => unreachable!(),
			};

			// Execute the request
			tracing::debug!("BedrockStreamer: Sending HTTP request...");
			tracing::debug!("BedrockStreamer: Creating send() future");
			let fut = builder.send();
			tracing::debug!("BedrockStreamer: Pinning future");
			tokio::pin!(fut);
			tracing::debug!("BedrockStreamer: About to poll future with context");

			match fut.poll(cx) {
				Poll::Ready(Ok(response)) => {
					let status = response.status();
					tracing::debug!("BedrockStreamer: Received response with status: {}", status);
					if !status.is_success() {
						tracing::error!("BedrockStreamer: Non-success status code: {}", status);
					}
					self.state = StreamerState::Streaming { response, done: false };
					// Fall through to polling
				}
				Poll::Ready(Err(e)) => {
					tracing::error!("BedrockStreamer: Failed to send request: {:?}", e);
					tracing::error!("BedrockStreamer: Error source: {:?}", StdError::source(&e));
					if e.is_timeout() {
						tracing::error!("BedrockStreamer: Request timed out");
					}
					if e.is_connect() {
						tracing::error!("BedrockStreamer: Connection error");
					}
					return Poll::Ready(Some(Err(Error::WebStream {
						model_iden,
						cause: format!("Failed to send request: {}", e),
					})));
				}
				Poll::Pending => {
					tracing::debug!("BedrockStreamer: Request still pending, will be polled again");
					tracing::debug!("BedrockStreamer: Waker registered, waiting for network event");
					return Poll::Pending;
				}
			}
		}

		// Poll the response stream - extract response to avoid borrow conflicts
		let (chunk_result, should_mark_done) = if let StreamerState::Streaming { response, done } = &mut self.state {
			if *done {
				return Poll::Ready(None);
			}

			// Poll the response body using async chunk reading
			let chunk_fut = response.chunk();
			tokio::pin!(chunk_fut);

			match chunk_fut.poll(cx) {
				Poll::Ready(Ok(Some(chunk))) => (Some(Ok(chunk)), false),
				Poll::Ready(Ok(None)) => (None, true),
				Poll::Ready(Err(e)) => {
					tracing::error!("Bedrock stream error: {}", e);
					(
						Some(Err(Error::WebStream {
							model_iden: model_iden.clone(),
							cause: format!("Reqwest error: {}", e),
						})),
						true,
					)
				}
				Poll::Pending => return Poll::Pending,
			}
		} else {
			return Poll::Ready(None);
		};

		// Mark as done if needed
		if should_mark_done {
			if let StreamerState::Streaming { done, .. } = &mut self.state {
				*done = true;
			}
		}

		// Process the chunk result
		match chunk_result {
			Some(Ok(chunk)) => {
				// Parse AWS event stream format
				match self.parse_aws_event(chunk) {
					Ok(events) => {
						// Return the first event, queue others if needed
						if let Some(event) = events.into_iter().next() {
							Poll::Ready(Some(Ok(event)))
						} else {
							// No events produced, continue polling
							cx.waker().wake_by_ref();
							Poll::Pending
						}
					}
					Err(e) => Poll::Ready(Some(Err(e))),
				}
			}
			Some(Err(e)) => Poll::Ready(Some(Err(e))),
			None => Poll::Ready(None),
		}
	}
}

# 테스트 목록 (한국어 정리)

이 문서는 `tests/` 디렉토리 안의 모든 테스트가 **무엇을 검증하는지**를 한국어로 정리한 것이다.
테스트가 추가/변경/삭제될 때마다 이 파일도 함께 갱신해야 한다 (CLAUDE.md 지침 참조).

## 파일 개요

| 파일 | 분류 | 사용 도구 |
|------|------|-----------|
| `test_config.py` | config 로딩 / 헤더 템플릿 단위 테스트 | yaml + tempfile |
| `test_router.py` | Router · 서킷 브레이커 단위 테스트 | AsyncMock httpx |
| `test_failover.py` | 페일오버 / 상태코드 / 직접요청 매트릭스 | AsyncMock httpx |
| `test_server.py` | FastAPI 통합 테스트 | TestClient + respx |
| `test_e2e.py` | 실제 HTTP end-to-end | mock_upstream (uvicorn) |
| `test_anthropic_adapter.py` | Anthropic 어댑터 변환 함수 단위 테스트 | (순수 함수) |
| `test_anthropic_e2e.py` | Anthropic 어댑터 e2e | mock_upstream (uvicorn) |

---

## test_config.py — 설정 로딩과 헤더 템플릿

### `resolve_headers` (헤더 템플릿 치환)
- **test_uuid_template_generates_value** — `{{uuid}}` 템플릿이 36자(하이픈 4개) UUID로 치환되는지.
- **test_uuid_template_unique_per_call** — 매 호출마다 다른 UUID가 생성되는지.
- **test_env_template_resolved** — `{{env:VAR}}` 가 실제 환경변수 값으로 치환되는지.
- **test_env_template_missing_var** — 환경변수가 없으면 빈 문자열이 되고 경고 로그가 남는지.
- **test_static_header_unchanged** — 템플릿 없는 일반 문자열은 그대로 통과하는지.
- **test_multiple_templates_in_one_value** — 한 값 안에 템플릿이 섞여 있어도 정상 치환되는지.

### `load_config` (YAML 파일 로딩)
- **test_load_minimal_config** — 최소 설정(엔드포인트만)으로 로딩 성공, 기본값(port=8000) 적용 확인.
- **test_load_full_config** — 모든 옵션이 채워진 풀 설정이 정상 파싱되는지.
- **test_load_config_no_endpoints_raises** — 엔드포인트가 비어있으면 예외.
- **test_load_config_invalid_url_raises** — `ftp://` 같은 비-HTTP URL이면 예외.
- **test_load_config_env_key** — `auth.api_keys` 안에서도 `{{env:...}}` 치환이 동작.
- **test_load_config_missing_file** — 존재하지 않는 파일 경로에 대해 `FileNotFoundError`.
- **test_same_endpoint_appears_twice_in_chain** — 같은 엔드포인트가 한 체인에 두 번 등장(다른 모델 사용)할 수 있고, `EndpointState`(서킷 브레이커)는 공유됨.

---

## test_router.py — Router 단위 테스트

### 기본 라우팅
- **test_candidates_priority_order** — 우선순위(priority) 전략에서 체인 순서대로 후보가 반환.
- **test_unknown_route_returns_empty** — 등록되지 않은 라우트 이름은 빈 리스트.
- **test_candidates_latency_strategy** — `latency` 전략이면 평균 지연이 낮은 엔드포인트가 우선.

### 서킷 브레이커 상태 전이
- **test_circuit_opens_after_threshold** — 연속 실패가 임계치에 도달하면 `closed → open`.
- **test_circuit_excludes_open_endpoint** — `open` 상태인 엔드포인트는 후보에서 제외.
- **test_circuit_transitions_to_half_open_after_cooldown** — 쿨다운 경과 후 `open → half_open`.
- **test_circuit_closes_on_success_from_half_open** — `half_open`에서 성공하면 `closed`로 복귀, 실패 카운터 0.
- **test_circuit_reopens_on_failure_from_half_open** — `half_open`에서 다시 실패하면 즉시 `open`.

### 통계
- **test_record_success_updates_stats** — 성공 기록 시 요청 수·평균 지연 갱신.
- **test_record_failure_updates_stats** — 실패(timeout) 기록 시 실패 수·timeout 카운트·timeout_rate 갱신.

### 상태 조회
- **test_get_status_returns_all_endpoints** — `get_status()`가 모든 엔드포인트를 반환.

### `execute()` (모킹 httpx)
- **test_execute_returns_on_first_success** — 첫 엔드포인트가 200을 반환하면 즉시 성공 반환.
- **test_execute_failover_on_timeout** — 첫 엔드포인트 타임아웃 → 다음 엔드포인트로 페일오버해 성공.
- **test_execute_raises_when_all_fail** — 모든 엔드포인트가 실패하면 `AllEndpointsFailedError`.

---

## test_failover.py — 페일오버 / 에러 코드 / 직접 요청 종합

### `_should_failover` — 페일오버 트리거 판정
- **test_status_code** (parametrize) — 상태코드별 동작 확정:
  - 2xx(200, 201) → 페일오버 X
  - 4xx 일반(400, 401, 403, 404, 422) → 페일오버 X (요청 자체가 잘못됨)
  - 408 / 429 → 페일오버 O
  - 5xx(500, 502, 503, 504, 520, 529) → 페일오버 O

### `TestExecuteHttpxExceptions` — httpx 예외별 페일오버
- **test_failover_on_timeout_exception** — `httpx.TimeoutException`에서 다음 엔드포인트로 (is_timeout=True).
- **test_failover_on_connect_timeout** — `ConnectTimeout`도 timeout으로 분류.
- **test_failover_on_read_timeout** — `ReadTimeout` 페일오버.
- **test_failover_on_pool_timeout** — `PoolTimeout` 페일오버.
- **test_failover_on_connect_error** — `ConnectError`는 timeout이 아니지만 페일오버됨.
- **test_failover_on_read_error** — `ReadError` 페일오버.
- **test_failover_on_remote_protocol_error** — `RemoteProtocolError` 페일오버.

### `TestExecuteStatusCodeFailover` — 상태코드별 페일오버
- **test_failover_on_retriable_status** (parametrize: 500/502/503/504/429/408) — 모두 페일오버 발생, error_message에 `HTTP {code}` 포함.
- **test_no_failover_on_client_error** (parametrize: 400/401/403/404/422) — 페일오버 없이 첫 엔드포인트의 응답을 그대로 클라이언트에 반환.

### `TestMixedErrors` — 혼합 시나리오
- **test_timeout_then_503_then_success** — alpha(timeout) → beta(503) → gamma(200), 시도 3번.
- **test_connect_error_then_429_then_success** — ConnectError → 429 → 200.
- **test_all_endpoints_timeout** — 전부 타임아웃이면 `AllEndpointsFailedError`.
- **test_all_endpoints_return_5xx** — 500/502/503 모두 실패해도 동일 예외.
- **test_mixed_connection_and_http_errors** — ConnectError + 500 + ReadTimeout 혼합 실패.

### `TestMaxRetries` — 재시도 횟수 제한
- **test_max_retries_limits_attempts** — `max_retries=1`이면 시도 2회(초기+재시도1)에서 종료.
- **test_max_retries_zero_means_single_attempt** — 0이면 재시도 없이 1회만 시도.
- **test_max_retries_exceeds_endpoints** — 100을 줘도 엔드포인트 수(3)로 제한됨.

### `TestDirectRequests` — `endpoint/model` 직접 지정
- **test_direct_skips_circuit_breaker_recording** — 직접 요청 실패는 서킷 브레이커 통계에 기록되지 않음.
- **test_direct_success_no_stats** — 성공도 기록되지 않음.
- **test_direct_passes_through_5xx** — 503을 502로 변환하지 않고 그대로 전달.
- **test_direct_passes_through_408** — 408도 그대로 전달.
- **test_direct_passes_through_429** — 429도 그대로 전달.

### `TestCircuitBreakerDuringExecute` — execute 중 서킷 동작
- **test_circuit_opens_after_threshold_across_calls** — 여러 execute 호출에 걸쳐 누적 실패가 쌓이면 `open`.
- **test_open_circuit_skips_endpoint_in_route** — `open`인 엔드포인트는 라우트에서 제외, 다음 엔드포인트가 처리.
- **test_failure_stats_updated_on_failover** — 페일오버 도중 실패한 각 엔드포인트의 stats가 모두 갱신.

---

## test_server.py — FastAPI 통합 테스트 (respx)

### Health & 기본 라우트
- **test_health_endpoint** — `GET /health` → `{"status": "ok"}`.
- **test_root_redirects_to_dashboard** — `GET /`이 `/dashboard`로 리다이렉트(3xx).

### 인증 미들웨어
- **test_no_auth_required_when_not_configured** — `auth` 설정이 없으면 키 없이도 요청 통과.
- **test_auth_rejects_missing_key** — 키가 필요한데 헤더 없으면 401.
- **test_auth_rejects_wrong_key** — 잘못된 키는 401.
- **test_auth_accepts_valid_key** — 유효한 키는 200.

### 프록시 동작
- **test_proxy_forwards_response** — upstream JSON 응답이 그대로 클라이언트에 전달됨.
- **test_proxy_returns_502_when_all_fail** — 모든 시도가 timeout이면 502.

### Status / Stats API
- **test_api_status_returns_endpoint_list** — `GET /api/status` → 엔드포인트 리스트, 초기 `circuit_state="closed"`.
- **test_api_requests_returns_empty_initially** — 요청 로그가 비었을 때 `total=0`, `rows=[]`.

---

## test_e2e.py — 실제 HTTP end-to-end (mock_upstream)

### `TestE2EFailover` — alpha(timeout) / beta(503) / gamma(ok) 체인
- **test_failover_timeout_then_503_then_success** — 200 응답을 받고, gamma가 응답 주체.
- **test_request_log_records_all_attempts** — 요청 로그에 attempts 3개 기록 (alpha=timeout, beta=503, gamma=success).
- **test_streaming_failover** — 스트리밍 요청도 정상 페일오버, `text/event-stream` 응답.
- **test_streaming_request_logged** — 스트리밍 요청도 SQLite 로그에 `is_stream=True`로 기록 (회귀 가드: `byte_generator`의 finally 블록이 fire-and-forget로 로그를 누락했던 버그 방지).

### `TestE2EAllFail`
- **test_all_fail_returns_502** — 체인 전체가 실패하면 502.

### `TestE2EDirectRequest`
- **test_direct_to_healthy_endpoint** — `gamma/mock-model` 직접 지정 → 200.
- **test_direct_to_error_endpoint_passes_through** — `beta/mock-model`(503) 직접 → 502 아닌 503 그대로.

### `TestE2EEndpointStatus` — 라우터 상태 보고
- **test_status_after_failover** — 페일오버 후 alpha total_failures/timeouts 증가, gamma는 성공 카운트.
- **test_health_endpoint** — `/health` 200 OK.

### `TestE2EErrorServer` — 추가 상태코드 검증
- **test_failover_on_408** — 408 반환 endpoint → 다음으로 페일오버, 200.
- **test_failover_on_429** — 429 반환 endpoint → 다음으로 페일오버, 200.

---

## test_anthropic_adapter.py — Anthropic 어댑터 단위 테스트

### `TestTranslateRequest` — Anthropic → OpenAI 요청 변환
- **test_basic_text_message** — 기본 텍스트 메시지 통과.
- **test_system_string_prepended** — 문자열 system → messages 맨 앞에 system role로 삽입.
- **test_system_array_concatenated** — system 배열의 text 블록들을 이어붙여 하나의 system 메시지로.
- **test_stop_sequences_renamed** — `stop_sequences` → `stop`.
- **test_passthrough_fields** — temperature/top_p/stream 그대로 전달.
- **test_top_k_dropped** — OpenAI 미지원, 삭제됨.
- **test_metadata_dropped** — metadata 삭제.
- **test_thinking_raises** — `thinking` 필드는 V1 미지원, ValueError.
- **test_tool_definitions_translated** — Anthropic tool 정의 → OpenAI `{type:"function", function:{name, description, parameters}}` 형식.
- **test_tool_choice_auto** — `{type:"auto"}` → `"auto"`.
- **test_tool_choice_any_maps_to_required** — `{type:"any"}` → `"required"`.
- **test_tool_choice_specific_tool** — `{type:"tool", name:"fn"}` → `{type:"function", function:{name:"fn"}}`.
- **test_tool_use_in_assistant_message** — assistant content 안의 `tool_use` → `tool_calls[]`로 이동.
- **test_tool_result_expands_to_tool_messages** — user의 `tool_result` 블록들을 별도의 `role:"tool"` 메시지들로 분리.
- **test_tool_result_with_content_blocks** — `tool_result.content`가 블록 배열이어도 텍스트 추출.
- **test_mixed_user_content_with_tool_result** — text + tool_result 혼합 시 user 메시지 + tool 메시지로 분리.
- **test_image_base64_translated** — base64 image → `data:mime;base64,...` URL.
- **test_image_url_passthrough** — URL image → OpenAI image_url 그대로.
- **test_assistant_tool_use_only** — tool_use만 있는 assistant 메시지는 `content=None`.

### `TestTranslateResponse` — OpenAI → Anthropic 응답 변환
- **test_text_content_wrapped_in_block** — 문자열 content → `[{type:"text", text:"..."}]`.
- **test_tool_calls_become_tool_use_blocks** — tool_calls → `tool_use` 블록 (arguments JSON 파싱).
- **test_mixed_text_and_tool_calls** — 텍스트 + tool_calls 동시 있을 때 두 블록 모두 생성.
- **test_finish_reason_stop_maps_to_end_turn** — `stop` → `end_turn`.
- **test_finish_reason_length_maps_to_max_tokens** — `length` → `max_tokens`.
- **test_finish_reason_tool_calls_maps_to_tool_use** — `tool_calls` → `tool_use`.
- **test_usage_fields_renamed** — `prompt_tokens` → `input_tokens`, `completion_tokens` → `output_tokens`.
- **test_id_gets_msg_prefix** — id 앞에 `msg_` 접두어.
- **test_model_uses_original** — 응답 model 필드는 클라이언트가 보낸 원본 모델명 사용.
- **test_empty_choices** — choices가 빈 배열이면 빈 content/`end_turn`.

### `TestSSEBuffer` — SSE 청크 버퍼 파싱
- **test_single_complete_event** — 한 번에 완성된 이벤트 1개 반환.
- **test_split_across_chunks** — 여러 청크에 걸쳐 와도 끝의 빈 줄을 만나야 이벤트 완성.
- **test_multiple_events_in_one_chunk** — 한 청크에 이벤트 여러 개 들어있으면 모두 분리.
- **test_done_event** — `[DONE]` 이벤트는 `_parse_sse_data`가 None 반환.

### `TestParseSSEData` — SSE data 라인 파싱
- **test_normal_json** — 정상 JSON 파싱.
- **test_done** — `[DONE]` → None.
- **test_event_with_named_type** — `event:` 라인이 앞에 있어도 `data:`만 추출.
- **test_invalid_json** — JSON 아니면 None.
- **test_no_data_line** — `event:`만 있으면 None.

---

## test_anthropic_e2e.py — Anthropic 어댑터 end-to-end

### `TestAnthropicNonStreaming` — 비스트리밍
- **test_basic_text_response** — `/v1/messages` POST → 200, Anthropic 응답 형식 (`type:"message"`, `content:[{type:"text"}]`, `stop_reason`, `usage` 필드).
- **test_system_prompt_forwarded** — system 프롬프트가 정상 처리되어 200 응답.
- **test_response_has_msg_id_prefix** — id가 `msg_` 접두어로 시작.
- **test_failover_works** — alpha(timeout) → beta(503) → gamma(ok) 페일오버 후 200.
- **test_request_logged** — 요청이 SQLite 로그에 success로 기록됨.

### `TestAnthropicDirect` — `endpoint/model` 직접 지정
- **test_direct_to_healthy_endpoint** — `gamma/mock-model` 직접 → 200, gamma 응답.
- **test_direct_to_error_returns_anthropic_error** — `beta/mock-model`(503) 직접 → 503 + Anthropic error 형식 (`type:"error"`, `error.type:"api_error"`).

### `TestAnthropicErrors` — 에러 응답 형식
- **test_unknown_model_returns_anthropic_404** — 존재하지 않는 라우트 → 404 + `error.type:"not_found_error"`.
- **test_invalid_json_returns_anthropic_error** — JSON 파싱 실패 → 400 + Anthropic error 형식.
- **test_all_fail_returns_anthropic_502** — 체인 전체 실패 → 502 + `error.type:"api_error"`.

### `TestAnthropicAuth` — `/v1/messages` 인증 미들웨어
- **test_rejects_missing_key** — Authorization 헤더 없으면 401.
- **test_rejects_wrong_key** — 잘못된 키는 401.
- **test_accepts_valid_key** — 유효 키는 200 + Anthropic message 형식 응답.

### `TestAnthropicStreaming` — 스트리밍
- **test_streaming_returns_sse** — `stream:true` → 200, `text/event-stream` 응답.
- **test_streaming_event_sequence** — 이벤트 **순서** 검증: `message_start`이 첫 번째, `message_stop`이 마지막, `message_delta`가 그 직전. 컨텐츠 블록은 `start → delta → stop` 순. (Anthropic 스펙 위반 시 클라이언트가 깨질 수 있으므로 순서 강제.)
- **test_streaming_contains_text** — gamma의 응답 텍스트가 스트림 내용에 포함.
- **test_streaming_request_logged** — 스트리밍 요청도 SQLite 로그에 `is_stream=True`로 기록 (회귀 가드: `wrapped_generator`의 finally 블록이 fire-and-forget로 로그를 누락했던 버그 방지).
- **test_streaming_message_start_has_model** — `message_start` data에 `model`(원본 모델명), `role:"assistant"` 포함.
- **test_streaming_failover** — 스트리밍에서도 페일오버 동작 (gamma 응답).
- **test_streaming_direct_to_healthy** — 직접 지정 스트리밍 동작.
- **test_streaming_all_fail_returns_error** — 스트리밍 중 모든 step 실패 → 502 + Anthropic error JSON.

### `TestOpenAIRegressionFromAnthropicTests` — OpenAI 회귀 검증
- **test_openai_chat_still_works** — `/v1/messages` 추가가 기존 `/v1/chat/completions`를 망가뜨리지 않았는지 확인.

# LLM Proxy

여러 OpenAI-compatible LLM API 엔드포인트에 자동 장애 조치(failover), 서킷 브레이커, 내장 대시보드를 제공하는 경량 로컬 프록시 서버입니다. OpenAI와 Anthropic 클라이언트 형식을 모두 지원합니다 — `POST /v1/chat/completions` (OpenAI) 또는 `POST /v1/messages` (Anthropic) 어느 쪽이든 동일한 upstream 엔드포인트에 요청할 수 있습니다.

## 주요 기능

- **Named route chains** — `alpha/gpt-4 -> beta/llama-3 -> gamma/mistral-7b`와 같은 fallback 체인을 정의; timeout/error 시 자동으로 다음 단계 시도
- **Direct addressing** — `endpoint_name/model_id` 형식으로 특정 엔드포인트에 직접 전송 (라우팅 및 서킷 브레이커 우회)
- **서킷 브레이커** — N회 연속 실패 후 해당 엔드포인트를 cooldown 기간 동안 제외, half-open probe로 자동 복구
- **Anthropic Messages API** — `POST /v1/messages` 엔드포인트가 Anthropic 형식을 OpenAI 형식으로 실시간 변환; 텍스트, tool use, 이미지, 스트리밍, 에러 형식 변환 지원
- **스트리밍** — OpenAI와 Anthropic SSE 형식 모두 지원하는 완전한 SSE / `stream: true` 지원
- **대시보드 GUI** — 실시간 엔드포인트 상태 모니터링, 요청 로그, 드래그 앤 드롭 라우팅 편집기
- **모델 자동 탐색** — 시작 시 모든 엔드포인트에서 사용 가능한 모델을 자동 감지
- **커스텀 헤더** — 엔드포인트별 정적 헤더, `{{uuid}}` 및 `{{env:VAR}}` 템플릿 지원
- **SQLite 로깅** — WAL 모드, 외부 의존성 없음
- **인증** — 선택적 API 키 보호
- **pip 설치 가능** — `pip install` + `llm-proxy start` 한 줄로 실행

## 시작하기

```bash
git clone https://github.com/JeongJuHwan-jip/llm-proxy.git
cd llm-proxy

# uv 설치 (미설치 시)
pip install uv

# 가상환경 생성 및 의존성 설치
uv venv
uv pip install -e ".[test]"

# 가상환경 활성화
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / Mac

# 예제 설정 복사 후 편집
cp config.example.yaml config.yaml
# config.yaml 편집 — 엔드포인트 URL과 헤더 설정

# 설정 검증
llm-proxy validate --config config.yaml

# 모델 탐색
llm-proxy discover --config config.yaml

# 서버 시작
llm-proxy start --config config.yaml
```

기본 리스닝 주소: `http://0.0.0.0:8000`
대시보드: `http://localhost:8000/dashboard/index.html`

## 대시보드

대시보드는 두 개의 탭으로 구성됩니다:

- **Monitor** — SSE로 실시간 업데이트되는 엔드포인트 상태 카드(서킷 상태, 지연 시간, 실패율)와 요청 이력 테이블
- **Settings** — 장애 조치 파라미터 설정 및 드래그 앤 드롭으로 라우팅 체인 구성

**Apply Settings** 클릭 시 `settings.json` 파일이 config.yaml 옆에 생성됩니다. 수동 파일 편집 불필요 — 대시보드가 전부 관리합니다.

## 설정 레퍼런스

`config.yaml`은 엔드포인트와 인프라를 정의합니다. 라우팅과 장애 조치는 대시보드에서 관리합니다.

```yaml
proxy:
  host: "0.0.0.0"
  port: 8000
  header_priority: "config"   # "config" 또는 "client" — 헤더 충돌 시 우선순위

endpoints:
  - name: "alpha"
    url: "https://alpha.company.com/v1"
    timeout_ms: 5000
    headers:
      X-Request-ID: "{{uuid}}"
      Authorization: "Bearer {{env:ALPHA_TOKEN}}"

  - name: "beta"
    url: "https://beta.company.com/v1"
    timeout_ms: 8000

logging:
  db_path: "./data/proxy.db"
  log_request_body: false

# auth:
#   api_keys:
#     - "{{env:PROXY_API_KEY}}"
```

### 헤더 템플릿

| 템플릿 | 변환 결과 |
|---|---|
| `{{uuid}}` | 매 요청마다 새로운 `uuid4()` 문자열 |
| `{{env:VAR_NAME}}` | `VAR_NAME` 환경변수 값 |

### 요청 라우팅

클라이언트는 요청 body의 `model` 필드로 라우팅을 결정합니다:

| model 값 | 동작 |
|---|---|
| `alpha/gpt-4` | **Direct** — 엔드포인트 `alpha`에 모델 `gpt-4`로 직접 전송. 서킷 브레이커와 장애 조치 없음. |
| `best-available` | **Named route** — 대시보드에서 정의한 fallback 체인을 따름 (예: alpha/gpt-4 -> beta/llama-3 -> gamma/mistral). |

## CLI

```
llm-proxy --help
llm-proxy start    --config config.yaml [--host HOST] [--port PORT] [--workers N]
llm-proxy validate --config config.yaml
llm-proxy discover --config config.yaml [--snippet]
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/v1/chat/completions` | 장애 조치 프록시 (OpenAI 형식) |
| POST | `/v1/messages` | 장애 조치 프록시 (Anthropic 형식) |
| GET | `/v1/models` | 통합 모델 목록 (direct + named routes) |
| GET | `/api/status` | 엔드포인트 상태 스냅샷 |
| GET | `/api/requests?limit=50` | 최근 요청 로그 |
| GET | `/api/discovery` | 모델 탐색 결과 + diff |
| GET | `/api/config/settings` | 현재 장애 조치 + 라우팅 설정 |
| PUT | `/api/config/settings` | 장애 조치 + 라우팅 적용 (settings.json에 저장) |
| POST | `/api/routing/reload` | 재시작 없이 settings.json 다시 읽기 |
| GET | `/api/events` | 대시보드용 SSE 스트림 |

## Anthropic Messages API 지원

`config.yaml`에 등록된 upstream은 모두 OpenAI-compatible API입니다. Anthropic adapter가 요청/응답 형식을 자동으로 변환합니다:

```
Anthropic 클라이언트 → POST /v1/messages
  → translate_request() [Anthropic → OpenAI 변환]
  → Router.execute() (기존 failover/서킷 브레이커 그대로 활용)
  → translate_response() [OpenAI → Anthropic 변환]
  → Anthropic 형식 응답
```

**지원 범위:**
- 텍스트 메시지 (system, user, assistant)
- 스트리밍 / 비스트리밍
- Tool use (custom tools)
- 이미지 (base64, URL)
- 에러 응답 형식 변환

**미지원 (향후 확장):**
- `thinking` / extended thinking
- Server tools (web_search, code_execution)
- Documents / PDFs
- Citations, cache_control

## Mock 서버를 이용한 로컬 테스트

```bash
# Mock LLM 서버 시작
python scripts/mock_llm_server.py --port 8001 --name alpha --models gpt-4,claude-3
python scripts/mock_llm_server.py --port 8002 --name beta --models gpt-4,llama-3 --behavior flaky --fail-rate 0.5
python scripts/mock_llm_server.py --port 8003 --name gamma --models mistral-7b

# Mock 서버 옵션:
#   --behavior ok|timeout|error|flaky|slow
#   --fail-rate 0.5          (flaky 모드)
#   --timeout-rate 0.3       (flaky 실패 중 timeout 비율)
#   --delay 5.0              (slow/timeout 지연 시간, 초)
#   --latency-min 0.1 --latency-max 0.8   (성공 시 랜덤 지연)

# 프록시 시작
llm-proxy start --config config.local-test.yaml
```

## 아키텍처

```
client (OpenAI 형식)       -> POST /v1/chat/completions  ----+
client (Anthropic 형식)    -> POST /v1/messages               |
                                |                              |
                                v                              |
                         Anthropic Adapter                     |
                    (translate_request: Anthropic→OpenAI)      |
                                |                              |
                                +------------------------------+
                                |
                                v
                         Router.execute(steps=[...])
                                |   +------------------------------------+
                                +-->| EndpointState (alpha)  [CLOSED]     | --> upstream (OpenAI-compatible)
                                |   |   circuit breaker, latency samples  |
                                |   +------------------------------------+
                                |   +------------------------------------+
                                +-->| EndpointState (beta)   [CLOSED]     | --> upstream (failover)
                                |   +------------------------------------+
                                |   +------------------------------------+
                                +-->| EndpointState (gamma)  [CLOSED]     | --> upstream (failover)
                                    +------------------------------------+
                                                |
                                                v
                                   translate_response (Anthropic인 경우)
                                                |
                                                v
                                        Database (SQLite WAL)
```

**파일 구성:**
- `config.yaml` — 엔드포인트 정의, 인증, 프록시 서버 설정 (수동 편집)
- `settings.json` — 장애 조치 설정 + 라우트 체인 (대시보드에서 관리, 자동 생성)
- `src/llm_proxy/adapters/anthropic.py` — Anthropic ↔ OpenAI 형식 변환 어댑터

## 개발

```bash
# 테스트 실행
pytest

# Mock 서버와 함께 실행
llm-proxy start --config config.local-test.yaml --port 9000
```

## Docker

```bash
docker build -t llm-proxy .
docker run -p 8000:8000 -v $(pwd)/config.yaml:/app/config.yaml llm-proxy
```

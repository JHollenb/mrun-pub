# Configuration

Configuration is explicit. mrun never reads ancestor .env files or searches sibling
repositories. No private fleet endpoint is built in.

| Variable | Default / purpose |
| --- | --- |
| MRUN_URL | http://127.0.0.1:9025; client scheduler endpoint |
| MRUN_BIND / MRUN_PORT | 127.0.0.1 / 9025; scheduler listener |
| MRUN_TOKEN | Required scheduler/client credential |
| MRUN_AGENT_TOKEN | Independent agent authority for guarded leases |
| MRUN_ALLOW_UNAUTHENTICATED | Explicit local-development opt-in; use only on loopback |
| MRUN_SERVER_DATA | ~/.mrun/server; queue, logs and SQLite records |
| MRUN_AGENT_CONFIG | ~/.mrun/agent.json; explicit agent JSON |
| MRUN_CACHE_ROOT | $XDG_CACHE_HOME/mrun or ~/.cache/mrun |
| MRUN_DATA_ROOT | $XDG_DATA_HOME/mrun or ~/.local/share/mrun |
| MRUN_MODELS_ROOT | <cache>/models |
| MRUN_STORES_ROOT | <models>/qstores |
| MRUN_ARTIFACT_ROOT | <data>/artifacts |
| MRUN_CLAIMS_DIR | ~/.local/state/mrun/claims |
| LLM_MODELS_EXTRA_ROOT | Explicit, path-separator-delimited additional materialized model roots |

An agent can run installed Python commands directly. Environment aliases are
optional operator-configured project locations, not discovered checkout paths.
Example agent JSON:

```json
{"server_url":"http://127.0.0.1:9025","model_roots":["/srv/models"],"max_concurrent":1}
```

Supply credentials through the process/service environment or an explicitly
permissioned agent configuration. Payload children never receive the general client
or agent token; debugger-capable jobs receive independently scoped credentials.

Use authenticated transport and an appropriate network boundary for remote fleets.
Linux cgroup containment and optional bubblewrap filesystem/process isolation remain
explicit capabilities. Bubblewrap shares host networking; macOS sampled containment
does not provide the Linux namespace boundary. See the contracts in payload_sandbox.py
and agent/config.py before enabling required sandbox jobs.

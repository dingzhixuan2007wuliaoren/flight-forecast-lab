# Strict fare provider recovery

The strict-fare chain never promotes search-only, timetable, model, or
unverified data into bookable results. Provider recovery changes availability,
not the evidence boundary.

## Failure classes

| Evidence | Action | Paid/search retry |
| --- | --- | --- |
| SerpApi Account API 401/403 or non-active account | Open the authentication circuit and expose a sanitized reason | No |
| SerpApi account becomes active | Close the circuit after the free Account API probe | Normal search resumes |
| Scrappa booking 429/500/502/503 or transport failure | Reserve one additional monthly credit, wait briefly, retry that booking token once | At most one |
| Scrappa 400/401/402/403/422 or strict evidence rejection | Stop that candidate | No |
| Ignav 424/503 or transport failure | Retry that request once under the lifetime quota wall | At most one |
| Ignav 400/401/402/403/404/429 or strict evidence rejection | Stop that candidate | No |

Every retry is candidate-scoped. A failed booking sweep is never replayed as a
whole-provider search. Calls are reserved before network I/O and included in
the response accounting.

## Credential rotation

Credentials remain Render environment secrets and must not be committed. A
SerpApi request always begins with the provider's free Account API. Invalid or
inactive credentials therefore cannot consume Google Flights searches. After
the five-minute circuit window, the next comparison starts with the free
Account API; it stops there if the account is still invalid, and continues with
normal searches only after the account is active. This lets an account recover
without clearing application state. Changing a Render secret also restarts the
service with a fresh provider instance.

Production sets `SERPAPI_CREDENTIAL_PREFLIGHT_ENABLED=1`. Each new service
instance therefore starts one background call to SerpApi's no-charge Account
API. The check never reserves fare quota, never delays `/health` or `/ready`, and
a transient Account API outage never makes readiness fail. Its result is reduced to
an allowlisted credential state, HTTP status, exception type, observation time,
and transient flag; keys, fingerprints, account identifiers, emails, response
bodies, and quota payloads are never returned or logged.

## Browser and proxy cache boundary

The dashboard stores only the route form and language in versioned session
state. It never persists or restores a comparison response, so a deployment,
refresh, or new search cannot replay a previous authentication or upstream
failure. The former `flight-forecast-session-v1` entry is deleted on page load.
Comparison requests use `cache: "no-store"`, and `/health`, `/ready`, `/version`,
and every `/v1/*` response carry `Cache-Control: no-store`.

## What cannot be repaired locally

No application can create a valid third-party credential, make an upstream
inventory source available, or turn an itinerary without a safe booking path
into a bookable fare. When all independent strict providers are unavailable,
the correct result is a sanitized temporary-unavailability status and zero
invented flights.

The local SQLite quota and circuit state is durable only for the lifetime of
the current filesystem. A production deployment that must preserve these
states across replacement instances needs an external durable database; Render
free web-service storage is not a permanent store.

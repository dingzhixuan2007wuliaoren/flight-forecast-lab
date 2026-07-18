# Runtime context and free-data fallbacks

Flight Forecast Lab can enrich its synthetic demonstration models with short-lived external
context. External data is never presented as part of the training corpus and is never silently
substituted for a live bookable fare.

## Resolution order

| Signal | Preferred source | Fallback | Meaning in the response |
| --- | --- | --- | --- |
| Weather | Open-Meteo current/forecast plus NOAA METAR/TAF | Clearly labelled synthetic month/latitude prior for display only; the on-time prediction switches to the weather-free model | `live`, `forecast`, or `proxy`; `weather_feature_status` says `used` or `ignored` |
| Airport operations | FAA NAS Status for authoritative current US events; optional AirLabs schedule samples elsewhere | ADSB.lol current aircraft-density proxy, plus a clearly labelled training/synthetic target-time prior | `live` or `proxy` |
| Disruption news | No-key GDELT DOC 2.0 articles from the recent seven-day window, ordered `DateDesc`; official GAL rolling RSS if DOC fails | Route cache no older than six hours with reduced influence, otherwise a neutral value with no articles | `live`, `historical`, or `neutral` |
| Strict flight comparison | Every configured/released strict source runs: SerpApi, SearchAPI.io, and Ignav only after its two explicit release controls; each source requires its own second booking-option verification step | With two or more attempted sources the service returns `strict_fare_aggregate` plus per-source `provider_runs`; equivalent full itinerary-and-cabin rows keep the lowest verified fare. Quarantined and reference-only sources never substitute a flight | `strict_bookable_only`; retained rows use `bookability_status=booking_option_verified` and preserve exact source attribution |

The service catches provider timeouts, malformed payloads, quota exhaustion, and empty results.
Context signals may return a labelled fallback rather than fail the whole prediction; strict fare
search returns the strictly verified subset allowed by the free quota and provider response, or
labelled empty `offers`, and never substitutes an unpriced row. Set
`EXTERNAL_CONTEXT_ENABLED=0` for deterministic offline context development.

Strict comparison needs at least one supported strict-fare credential. The provider roles and local
free-only hard stops are:

| Provider | Runtime role | Local free ceiling and reset scope |
| --- | --- | --- |
| SerpApi Google Flights | First configured aggregating strict fare source | Up to 250 reserved requests per provider billing period; the boundary follows `plan_renewal_date`, not a calendar month |
| SearchAPI.io Google Flights | Aggregating strict fare source | 100 lifetime account requests from the signup allocation; it does not renew monthly |
| Ignav | `ignav_quarantine` by default; `ignav_verified_fares` only after release | Up to 1,000 lifetime requests; both `IGNAV_STRICT_RELEASE=1` and `IGNAV_FREE_ACCOUNT_ATTESTED=1`, plus controlled validation, are required before it can supply strict offers |
| Scrape.do Google Flights | Aggregate reference only | 1,000 credits per month; 10 credits are reserved per reference call, with at most one controlled retry for a transient failure |
| OpenSky Network | Current traffic-density reference only | Anonymous mode is active by default without credentials at 400 API credits per day; OAuth mode may use at most 4,000 per day |
| AeroDataBox | Dated schedule reference only | Trusted RapidAPI free-plan limit/remaining/reset headers define a provider cycle; without trustworthy reset evidence, one installation-lifetime 600-unit wall applies |
| AirLabs | Airport-operations and timetable reference only | One atomic cross-process SQLite monthly ledger is shared by every transport; an absent/invalid explicit local limit fails closed, and valid account counters can only tighten the wall |

Configuration, quota, quarantine, and evidence are separate states. `GET /v1/provider-status` never
returns credentials, request parameters, raw provider errors, booking tokens, or complete outbound
URLs. Reference-only and quarantined providers cannot populate `offers`, even when configured.
Ignav therefore remains `ignav_quarantine` in default and `auto` operation until both release
controls are true; a released result uses the distinct `ignav_verified_fares` provider code rather
than relabelling quarantined data.

SerpApi Free currently includes 250 successful searches per
provider billing period plus 50 requests per hour; the period boundary follows the account's
`plan_renewal_date`, not the calendar month. Initial Google Flights searches and `booking_token`
follow-ups share that allowance. The application applies one persistent local billing-period ledger
and a hard ceiling of 250 reserved attempts.
Configure the same shell that starts the server:

```powershell
$env:FLIGHT_OFFER_PROVIDER="auto"
$env:SERPAPI_API_KEY="your-serpapi-key"
$env:SERPAPI_MONTHLY_LIMIT="250"
$env:SEARCHAPI_API_KEY="your-searchapi-key"
$env:SEARCHAPI_LIFETIME_LIMIT="100"
$env:AIRLABS_API_KEY="your-free-key"
$env:AIRLABS_MONTHLY_CALL_LIMIT="1000"
$env:AIRLABS_USAGE_DB="runtime/airlabs-usage.sqlite3"
python -m flight_forecaster serve --model-dir artifacts/demo
```

The application reads the process environment and does not automatically load `.env`. Never put
credentials in source code, commits, browser/frontend data, or application logs. SerpApi officially
requires the server to include the key in the HTTPS query sent to `https://serpapi.com`, so complete
outbound provider URLs must not be logged.
`SERPAPI_MONTHLY_LIMIT` defaults to 250 when omitted. Values above 250 are clamped to 250, and an
invalid or non-positive value does not disable the safety ceiling. One comparison performs the four
SerpApi cabin searches and then processes every eligible `booking_token` candidate those searches
actually return. The number of booking-option validation calls therefore varies with the response
and remaining hourly/monthly free allowance. Results are bounded by SerpApi coverage, provider
responses, and the configured free quota; they are not a complete global list of airlines or flights.
SearchAPI's local wall is the one-time 100-request signup allocation for the installation/account;
it is never described or reset as a monthly allowance. Its free account endpoint can represent this
as `monthly_allowance=0` with the unused grant in `remaining_credits`; that zero is accepted only
when the subscription is absent and all returned counters remain within the 100-request free ceiling.
The independent SQLite lifetime ledger is still authoritative. `auto` invokes every configured and released
strict source, even after another source confirms offers. One actually attempted source keeps its own
provider identity; two or more attempted sources produce `strict_fare_aggregate` with non-nested
`provider_runs` containing each source's status, coverage, quota, and sanitized diagnostics. Only
offers independently verified by their own source can be merged. A source's failure, processing, or
quota truncation remains visible in its run and in aggregate coverage; all completed sources must lack
a verified offer before the aggregate can be a complete `no_results`.

The AirLabs key remains optional and is used for airport-operations samples and timetable
references, not to make strict offers. When a key is present, `AIRLABS_MONTHLY_CALL_LIMIT` must be a
valid positive value or AirLabs fails closed before network I/O. Context and timetable calls share
one atomic cross-process SQLite ledger and reserve one call before every transport. Valid response
`request.key.limits_by_month` / `limits_total` counters are persisted without account identifiers
and may only reduce the later local allowance; they never expand it. Missing credentials or exhausted quota are supported modes:
context signals may still use labelled model/proxy fallbacks, while strict flight comparison
returns a structured empty result instead of inventing flights.

### Scrape.do reference boundary

Scrape.do is queried only as an aggregate, economy-cabin coverage reference after the strict fare
pool has produced no usable offer. It never creates an offer, booking URL, ranking entry, or offer
detail page. The adapter discards booking tokens, seller URLs, individual itinerary facts, and raw
error bodies. Its response contains only bounded candidate counts, optional aggregate price facts,
and sanitized HTTP/exception metadata.

Every uncached Google Flights plugin attempt reserves 10 credits in a durable local ledger. Before
that reservation, `GET https://api.scrape.do/info` must confirm `IsActive=true`,
`MaxMonthlyRequest<=1000`, and `RemainingMonthlyRequest>=10`; this prevents a paid-capacity account
or an exhausted free account from being used. The adapter also parses the provider's
`Scrape.do-Request-Cost` and `Scrape.do-Remaining-Credits` response headers, while never exposing
the token. A transient transport, 408/425/429, or 5xx failure may be retried exactly once, with a
fresh account check and a second 10-credit reservation. Cache hits consume no credits.

### OpenSky and AeroDataBox reference boundaries

OpenSky is a current trajectory-density reference, not fare, inventory, delay, or timetable proof.
Anonymous mode works without credentials and is locally bounded to 400 daily API credits; any
OAuth failure falls back to the anonymous ceiling rather than silently retaining a larger limit.

AeroDataBox returns dated timetable references only. A route/date lookup pre-reserves the complete
bounded request-unit cost before either provider transport. The ledger accepts a new quota period
only from trustworthy RapidAPI free-plan limit, remaining, and reset headers; ambiguous generic
rate-limit headers and paid-capacity limits are ignored. Until such a trusted reset is persisted,
all calls share a conservative 600-unit installation-lifetime wall, which does not reset at a
calendar-month boundary. After a trusted provider reset passes, exactly one pre-reserved probe may
establish the next authenticated free-plan period. AeroDataBox data never enters `offers`.

Within two hours of departure, the service uses fresh Open-Meteo current model conditions and
can blend NOAA METAR airport observations with TAF. From two to 30 hours it blends the
departure-hour Open-Meteo forecast with TAF; after that, it uses Open-Meteo hourly forecasts up
to the provider's 16-day limit. Open-Meteo current conditions are based on 15-minute model data,
not an airport sensor observation. If Open-Meteo is unavailable near departure, a fresh METAR
can now serve as the independent `live` fallback. NOAA responses are cached for five minutes to
stay below the provider's per-thread frequency guidance. TAF risk is calculated only from decoded
forecast segments that cover the requested departure time; METAR risk also reads structured wind,
gust, visibility, flight category, and ceiling fields rather than relying on weather keywords alone.

The price model never uses weather. The on-time prediction uses the weather-aware model only when
the target-departure weather context is `live` or `forecast`; for a `proxy`, missing, or unusable
weather signal it switches to the weather-free model and returns
`weather_feature_status=ignored`. The dashboard and offer detail page then state that weather was
omitted from this on-time prediction.

Date-only comparison resolves weather at the documented origin-local reference time. A returned
offer is allowed to use that weather only when its actual departure is within two hours of the
reference. More distant flights use the weather-free model and expose
`offers[].weather_feature_status=ignored`, avoiding a noon forecast being applied across the day.

## Airport operations: current snapshot versus target signal

Airport operations deliberately have two layers in `POST /v1/compare`:

- `context.operations` is the target-departure signal actually passed to the on-time model;
- `context.operations.current_snapshot` is a separate request-time snapshot for explanation.

This prevents a current event from being silently applied to a flight weeks in the future. When a
provider event or schedule sample genuinely covers the requested departure, the top-level signal
may use it. Otherwise, the target signal remains a `proxy` based on an origin training average or
the synthetic airport-size/time-of-day prior, while the current snapshot can still be shown. The
response exposes `method`, `data_tier`, `applicability`, `metrics`, `events`, sample/window fields,
and `fallback_reason` so clients can distinguish these cases.

For airports identified as being in the United States, the service queries the no-key
[FAA NAS Status airport-events feed](https://nasstatus.faa.gov/api/airport-events) and caches the
global response for 90 seconds. Structured airport closures, ground stops, ground-delay programs,
arrival/departure delays, deicing, and airport configuration can appear in the current snapshot.
Only events whose stated interval covers the target departure are allowed into the target signal.
Unbounded events may apply only when departure is within 90 minutes. FAA `freeForm` notices are
treated as low-severity scoped restrictions and are never promoted into the target signal; a notice
that affects transient general aviation, for example, is not a full commercial-airport closure.
An airport absent from the FAA event list means only that no current FAA event was listed, not that
every flight is guaranteed to operate normally.

For other airports, an optional AirLabs key enables a schedule sample in a ±90-minute window around
now and, when the provider returns applicable rows, around the target departure. The score uses
reported departure delay and cancellation fields. The free request is capped at 50 rows; responses
at that limit are marked `sample_truncated`, so the result is a sample rather than a complete airport
performance statistic. Provider schedule horizons and quotas may further restrict availability.

If no actual current sample is available, ADSB.lol counts aircraft near the airport coordinates and
returns a `proxy` traffic-density snapshot. It is permitted to affect the target signal only when the
departure is within 90 minutes of the request. Aircraft density is not an actual delay rate,
cancellation rate, airport throughput measurement, or official ground-stop feed. For later
departures the target signal uses the labelled prior. The demo has no validated historical airport
aggregate, so a hand-built prior must not be presented as authoritative historical performance.

The comparison dashboard sends `departure_date`, not a flight clock time. The date cannot be before
today in the origin IANA timezone and cannot be more than 370 local calendar days ahead. A future
date uses origin-local noon as the model/context reference. For today, noon is retained when it is
more than 30 minutes after generation; otherwise the service advances 30 minutes on the UTC timeline
and converts that instant back to the origin timezone. A next-day result is rejected with 422. The
response labels these cases `origin_local_noon_model_reference` or
`origin_local_remaining_day_model_reference`. Neither timestamp is an actual flight departure. A
real flight number or clock time appears in the main list only when every segment of a
provider-priced offer passes route, date, timezone, continuity, duration, cabin, fare, and
future-departure validation. AirLabs rows may show those fields only inside the clearly separated
reference list.

Strict mode uses an evidence chain, not a schedule projection. An active strict adapter searches its
fare source for each of the four requested travel classes, then processes every eligible candidate
that carries the source's opaque booking-verification token or identifier, subject to the remaining
free quota and provider availability. Where supported, `deep_search=true` and `show_hidden=true`
broaden visible results but do not
guarantee all airlines, flights, cabins, sellers, or private fares. Each attempted candidate is
followed by a booking-options request. Only a response
whose `selected_flights` exactly matches the original segment sequence and contains a seller,
matching flight numbers, positive one-way USD price, and HTTPS `booking_request.url` can enter
`offers`. The itinerary must also remain continuous, use one provider-confirmed cabin throughout,
contain real flight numbers and complete local/UTC times, and have one to four segments (zero to
three stops). Its `live_fare` is independent of `estimated_price_usd` and its 80% model interval.
The free response does not reliably establish whether taxes are included, so `taxes_included` is
unknown rather than asserted true. After all configured strict sources run, equivalent offers are
grouped by the complete ordered segment identity and cabin (not merely validating airline or route).
Across sellers and sources, only the lowest final verified fare in each group is retained; the
retained row keeps the provider identity that supplied its evidence.

If a SerpApi response is `Processing` or `Queued`, the adapter validates the opaque Search ID
against an allowlist and polls only the fixed `https://serpapi.com/searches/{search_id}.json`
archive path with bounded 0.5, 1, 1.5, and 2 second backoff. It never follows the response's arbitrary
archive URL. Archive reads are reported as `archive_poll_count`; they do not increase `call_count` or
the conservative quota reservation. After an unresolved poll, provider search error, transport
failure, or HTTP 408/425/5xx, the adapter may resubmit that cabin-search or booking-options request
exactly once, and only after a new atomic quota reservation succeeds. The second submission may use
the same bounded polling path but cannot trigger a third submission. HTTP 400, authentication
failure, 429, and strict payload rejection are not retried. An unresolved second attempt returns
`provider_processing`; terminal/HTTP/transport failure returns `provider_error`; `no_results` is
reserved for a successfully completed search with no offer that passes strict verification. If the
retry is blocked by the quota wall, `retry_quota_limited`, `coverage_status`, and `quota_limit`
report that fact without relabelling the first submitted attempt as skipped.

The free allowance is a provider-account limit, not an unlimited service guarantee. The local
ledger conservatively reserves every attempted initial or token request before issuing it, and stops
before `SERPAPI_MONTHLY_LIMIT` is exceeded. Its compatibility field `monthly_calls_used` follows the
`plan_renewal_date` billing period and counts reserved attempts, not provider-billed successes.
SerpApi currently says cached, errored, and failed searches do not count against provider quota, but
the local ledger still reserves them and can stop early to guarantee the free ceiling. Review SerpApi
account usage independently because provider quota rules
and plan terms can change. Missing configuration, exhausted local budget, authentication failure,
rate limiting, provider failure, no matching offers, an absent `booking_token`, an itinerary
mismatch, or a missing usable booking option is represented by explicit
`fare_search_metadata.status` and `result_status` values; none enables a model or timetable
fallback. `fare_search_metadata` also exposes `coverage_scope`,
`eligible_candidate_count`, `verification_attempted_count`, `verified_candidate_count`,
`strictly_rejected_candidate_count`, `provider_failed_candidate_count`,
`search_failed_cabin_count`,
`quota_skipped_candidate_count`, `deduplicated_verified_count`, `coverage_status`, `quota_limit`,
and `retry_quota_limited`. For `strict_fare_aggregate`, these top-level counts are the sums across
attempted runs before cross-source deduplication, while `provider_runs` preserves every individual
source snapshot. `coverage_status` is `complete`, `quota_limited`, `provider_incomplete`,
`quota_and_provider_incomplete`, or `not_evaluated`; `quota_limit` identifies the hourly or monthly
ceiling when quota truncated validation. A response may therefore contain a useful partial set of
strictly verified offers while clearly reporting which eligible candidates were not verified.
On a five-minute strict-cache hit, per-response call counts are zero while the candidate coverage
counts describe the original search; `cache_hit=true` and the bilingual notice make that distinction
explicit.

Provider diagnostics are deliberately data-minimized. The response contains at most ten records,
and the ignored runtime SQLite database retains at most 500 across restarts. A record contains only
observation time, request stage, HTTP status, a stable exception type, and a format-validated Search
ID. API keys, booking tokens, request parameters, complete URLs, and raw provider error text are
never included.

An upstream fare provider may serve a cached result; the application also maintains
a five-minute strict-result cache. `live_fare.provider_cache_hit` is inferred from a local cache hit
or provider status/time heuristics; it detects likely reuse but does not identify or prove the exact
cache layer. `provider_cache_age_seconds` is the age of the
displayed result relative to `verified_at`, bounded to 0–3900 seconds to allow processing and clock
tolerance. `verified_at` is the provider result/verification time, not the current
API response time. The dashboard displays response generation time separately.

AirLabs `schedules` and `routes` are reference-only in strict comparison. A complete dated schedule
or recurring weekday projection may appear in `timetable_references` with
`bookability_status=unverified`, but neither can enter `offers` or rankings because AirLabs does not
confirm current fare and availability. Cancelled, departed, active, landed, and past live identities
continue to suppress matching recurring references, so a routes projection cannot revive them.
AirLabs calls still require `AIRLABS_API_KEY`, a valid explicit local call limit, and the shared
SQLite ledger; they remain constrained by free quota, row limits,
field completeness, time horizon, and route coverage. Terminal values are provider estimates, not
confirmed day-of-operation assignments. Possible terminals and last-used aircraft from recurring
rows are not returned as facts about the selected date. `schedule_observed_at` is the fetch time for
a live schedule but the route record's `updated` time for a recurring projection; it is not one
uniform freshness clock.

Each schedules/routes request is capped at 50 rows. If either endpoint reports `request.has_more`
or returns 50 rows, comparison and offer-detail responses set `schedule_sample_truncated=true` and
`schedule_sample_limit=50`; their bilingual warning/notice states that the AirLabs timetable
reference list may be incomplete. This flag says nothing about strict-fare provider coverage.
`false` means no truncation signal was observed from AirLabs endpoints actually queried; it does not
prove complete reference coverage when an endpoint was skipped, unavailable, quota-limited, or
outside its time window. The free AirLabs integration does not attempt pagination.

Strict mode never creates route-level, timetable-only, or model-only flight offers. Every retained
offer is a complete provider `priced_offer` with `cabin_status=provider_confirmed` and
`bookability_status=booking_option_verified`. Its source fields are provider-aware:
`serpapi_google_flights_booking`, `searchapi_google_flights_booking`, or
`ignav_verified_booking` only for explicitly released `ignav_verified_fares`; offer-detail legs use
the matching `serpapi_booking_confirmed`, `searchapi_booking_confirmed`, or
`ignav_verified_booking_confirmed` basis. `provider_direct` represents one segment;
`provider_itinerary` represents two to four continuous segments. `route_airlines`, model hubs,
catalogue cabin expansion, and unresolved O&D scenarios cannot populate the main list. The schema
retains older enum values for compatibility, but strict responses do not generate those offers.

This approach does not pretend to provide universal coverage. Querying and aggregating every
configured free strict source reduces dependence on a single provider but does not enumerate global
market inventory. Even with `deep_search` and
`show_hidden`, Google Flights and its visible sellers
may omit a route, airline, cabin, date, private fare, or booking option. Strict mode returns no offer
in those cases; it never fabricates a replacement from AirLabs, an airline catalogue, or the
prediction model. A verified booking option is a provider snapshot at the query time, not a
promise that the airline or seller will preserve the same inventory, fare rules, or final checkout
price.

## News feature

Recent news is queried without an API key through GDELT DOC 2.0, using the origin and destination
codes/names together with a narrow disruption vocabulary covering airport closure, strikes,
conflict, severe disruption, cancellations, and related events. The seven-day query requests
`DateDesc`, so results are processed newest-observed first. GDELT's global source stream is
near-real-time rather than instantaneous; its core data updates on an approximately 15-minute
cycle.

If the DOC request fails, the service tries GDELT's official Global Article List (GAL) RSS feed.
That feed updates every minute and contains a rolling window of roughly the latest 15 minutes.
Only route-relevant disruption titles are accepted from the global RSS stream. Successful results
are cached by route for 15 minutes, independent of the requested departure date, which avoids
repeating the same external query and reduces rate-limit pressure. Concurrent requests for the
same route share a single refresh. If both live paths fail, a successful route cache no older than
six hours may be returned as `historical`; its risk contribution is reduced. With no usable cache,
the response is `neutral`, has value zero, and contains no articles.

Only real returned article titles are scored. The bounded score becomes `news_disruption_index`,
which is a feature in both demo models. Current news is less informative for departures farther in
the future, so its model influence is attenuated as lead time increases. Articles are deduplicated
by canonical URL and normalized title, and results prefer independent source domains instead of
counting syndicated copies as separate corroboration.

The main comparison response contains at most five articles. `POST /v1/context/news-detail` uses a
separate rich response with up to 20 articles and exposes the original title, source domain,
optional source language, GDELT indexed time, conservative category, matched risk terms, raw score,
recency factor, and weighted score. The detail route risk is attenuated according to how far away
the requested departure is. Headlines remain in their source language and are not machine
translated.

This is deliberately conservative:

- no article is invented when GDELT and the bounded stale cache are unavailable;
- article presence does not prove that a particular flight will be affected;
- the UI exposes article links, source, optional source language, observed/indexed time, and the
  signal status;
- GDELT's article time usually means when GDELT observed/indexed the URL, not a guaranteed exact
  publisher timestamp; headlines remain in their source language;
- the synthetic training relationship is an engineering demonstration, not validated causal
  evidence that news raises a particular fare or delay probability.

## Second-level offer, weather, and news pages

Every strict comparison offer links to `GET /details/offer`. The page calls
`POST /v1/offer-detail` with the route, `departure_date`, opaque `offer_id`, and `force_refresh`.
Initial load sends `force_refresh=false` and reuses the five-minute strict cache. Only the explicit
Refresh and re-query button sends `force_refresh=true`, which reruns the four cabin searches and
attempts every eligible returned candidate allowed by the remaining free quota and provider
response. The frontend timeout is 360 seconds so bounded batches of free-quota candidate
validation can finish. Its response repeats the selected
strict provider offer and returns the complete one-to-four segment itinerary, including flight numbers,
local/UTC times, confirmed cabin, booking/fare fields when supplied, and provider-confirmed layovers. AirLabs
timetable and model-only rows have no offer
ID or detail link; an expired or no-longer-confirmed ID returns 404. A confirmed price is still a
time-sensitive snapshot rather than a booking guarantee. The API exposes only an HTTPS booking URL
returned by the verified provider response; it never invents one.

The detail response also includes `price_curve`, a daily model projection from the current
origin-local date to the final honest pre-departure date. Each point contains a simulated quote
date/time, lead days, model estimate, and 80% interval. The first point uses the same generated time
and features as the selected offer. The final quote instant is always strictly before departure;
for a flight exactly at local midnight there is no falsely relabelled point on the departure date.
The entire curve is generated in one request by changing simulated quote time. It is explicitly not
collected fare history, a live fare, or a bookable quote, and points beyond the model's main 180-day
training horizon are marked as extrapolation.

The dashboard creates same-tab, shareable URLs for `GET /details/weather` and
`GET /details/news` after a successful comparison. Query parameters contain `origin`,
`destination`, canonical `departure_date`, `departure_time_basis`, and `lang`. Before navigation,
the dashboard stores the form, active ranking, language, and latest comparison in `sessionStorage`;
using Back restores that session state when the browser still has it. Each detail-page refresh sends
the date, not the previous fixed reference instant, so the API recomputes a safe noon or remaining-day
reference from that request's `generated_at`. Legacy clients may instead send only
`departure_time`; responses identify that path with `departure_time_basis=legacy_input`.

The weather page calls `POST /v1/context/weather-detail`. It shows both the origin at the
date-derived model/weather reference and destination at route-model-estimated arrival reference,
explicitly stating that neither is a flight schedule. It includes current Open-Meteo model
conditions, the nearest target-hour forecast, a ±12-hour hourly window (up to 25 points), weather
risk components, provider validity/fallback metadata, and available NOAA METAR/TAF raw reports with
conservative bilingual explanations. Separate `aviation_metadata` explains an empty report list
(no ICAO, no applicable report, or provider failure). If an applicable NOAA report has the highest
risk, the overall risk source and validity timestamps follow that report. Automated report interpretation is informational and does not
replace an official aviation-weather briefing. Open-Meteo and aviation responses are fresh-cached
for 10 minutes; a successful weather payload up to six hours old may be displayed as a clearly
labelled stale cache if refresh fails. The page has manual refresh and auto-refreshes every 10
minutes.

This weather detail is explanatory context. It is never an input to the price model, and it enters
the on-time prediction only when the comparison's departure weather is usable as `live` or
`forecast`. When `weather_feature_status=ignored`, the dashboard and offer detail page display the
bilingual notice that weather was omitted from this on-time prediction.

The news page calls `POST /v1/context/news-detail`. It shows the recent-seven-day article scoring
described above, a detail-page route raw-risk score, departure attenuation, and the exact main-context
news input in `model_effect` / `model_signal`, including that model signal's source and timestamp.
The larger detail article set is explanatory and is not mislabeled as the model input. Successful detail results are fresh-cached for 15 minutes; if live
refresh fails, a cache up to six hours old is labelled `historical` and its route risk is reduced by
50 percent. Without a usable cache, the result is `unavailable`, has zero risk, and contains no
invented articles. The page has manual refresh and auto-refreshes every 15 minutes.

Both pages use the dashboard's `中文 / English` switch. Manual or automatic refresh starts a new API
request, but a short-lived server cache may legitimately make the underlying provider snapshot
unchanged.

## Student, baggage, and fare-rule fields

The comparison response uses three-state or multi-state policy fields. `unknown` is not treated as
`not_included`. Ordinary free Google Flights queries cannot verify an actual student-only discount,
so the UI shows that criterion as `unknown` and it receives no ranking bonus. A public airline
student-program page is labelled `program_available`; it is not labelled as an actual student fare
for the requested trip. `confirmed_free`,
`confirmed_included`, and `confirmed_discount` are reserved for applicable offer-level evidence;
missing or ambiguous fare-rule fields remain `unknown`.

The student ranking follows the requested lexicographic order:

1. lower confirmed `live_fare.total_amount` (one adult, one way, USD; tax inclusion unknown);
2. confirmed free checked baggage;
3. confirmed actual student discount;
4. confirmed free change/refund;
5. lower age and verification burden.

`program_available` alone does not satisfy the actual-discount criterion. With the current strict
fare sources, actual student discount remains unknown and gives no score. Its published age and
verification metadata is considered only in the fifth criterion. The synthetic model estimate and
price curve never replace the confirmed live fare in ranking. Because confirmed totals will usually
differ, later criteria most often act as tie-breakers.

## Source links and operational limits

- [NOAA Aviation Weather Data API](https://aviationweather.gov/data/api/)
- [Open-Meteo forecast API](https://open-meteo.com/en/docs)
- [Open-Meteo free API terms and attribution](https://open-meteo.com/en/terms)
- [FAA NAS Status](https://nasstatus.faa.gov/)
- [SerpApi Google Flights API](https://serpapi.com/google-flights-api)
- [SerpApi Google Flights booking options](https://serpapi.com/google-flights-booking-options)
- [SerpApi plans, successful-search accounting, and free billing-period quota](https://serpapi.com/pricing)
- [AirLabs schedules API](https://airlabs.co/docs/schedules)
- [AirLabs routes API](https://airlabs.co/docs/routes)
- [ADSB.lol public API](https://www.adsb.lol/docs/open-data/api/)
- [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
- [GDELT Global Article List RSS](https://blog.gdeltproject.org/announcing-the-gdelt-article-list-rss-feed/)
- [GDELT project use and attribution](https://www.gdeltproject.org/about.html)
- [OurAirports public-domain data](https://ourairports.com/data/)

Free-provider quotas, fields, coverage, and terms can change. Review the linked sources before
deploying publicly or commercially. Use or redistribution of GDELT data must cite and link the
GDELT Project; the dashboard includes that visible attribution. API keys belong in the process
environment or another local secret store, never in the repository, browser/frontend, or application
logs. The server sends each credential only to its configured provider over HTTPS; complete
outbound URLs must not be logged. The application does not
automatically load `.env`; set provider variables in the shell that starts the server.

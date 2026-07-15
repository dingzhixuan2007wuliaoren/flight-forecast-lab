# Runtime context and free-data fallbacks

Flight Forecast Lab can enrich its synthetic demonstration models with short-lived external
context. External data is never presented as part of the training corpus and is never silently
substituted for a live bookable fare.

## Resolution order

| Signal | Preferred source | Fallback | Meaning in the response |
| --- | --- | --- | --- |
| Weather | Open-Meteo current/forecast plus NOAA METAR/TAF | Clearly labelled synthetic month/latitude prior | `live`, `forecast`, or `proxy` |
| Airport operations | FAA NAS Status for authoritative current US events; optional AirLabs schedule samples elsewhere | ADSB.lol current aircraft-density proxy, plus a clearly labelled training/synthetic target-time prior | `live` or `proxy` |
| Disruption news | No-key GDELT DOC 2.0 articles from the recent seven-day window, ordered `DateDesc`; official GAL rolling RSS if DOC fails | Route cache no older than six hours with reduced influence, otherwise a neutral value with no articles | `live`, `historical`, or `neutral` |
| Route airlines | AirLabs route records when a key is configured | Global connecting model-scenario catalog | `provider_confirmed` or `model_scenario` |

The service catches provider timeouts, malformed payloads, quota exhaustion, and empty results.
It then returns a labelled fallback rather than failing the whole prediction. Set
`EXTERNAL_CONTEXT_ENABLED=0` for deterministic offline development.

Within two hours of departure, the service uses fresh Open-Meteo current model conditions and
can blend NOAA METAR airport observations with TAF. From two to 30 hours it blends the
departure-hour Open-Meteo forecast with TAF; after that, it uses Open-Meteo hourly forecasts up
to the provider's 16-day limit. Open-Meteo current conditions are based on 15-minute model data,
not an airport sensor observation. If Open-Meteo is unavailable near departure, a fresh METAR
can now serve as the independent `live` fallback. NOAA responses are cached for five minutes to
stay below the provider's per-thread frequency guidance. TAF risk is calculated only from decoded
forecast segments that cover the requested departure time; METAR risk also reads structured wind,
gust, visibility, flight category, and ceiling fields rather than relying on weather keywords alone.

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

The dashboard sends a timezone-free wall time. The service resolves the origin airport from its
coordinates, determines the IANA timezone offline, handles daylight-saving offsets, and returns
both the normalized timestamp and timezone. API clients may alternatively send an aware ISO 8601
timestamp, which is treated as an absolute instant.

AirLabs route matches are treated as confirmed direct carriers. Every other catalog carrier is
kept as a one-stop model scenario so “direct first” never labels an unconfirmed route as direct.
Cabins always remain `catalog_scenario`, because the route source does not confirm cabin inventory.
For a one-stop scenario the displayed duration adds a 90-minute connection and the itinerary
on-time probability is the direct-leg probability squared. The response labels this assumption as
`two_leg_independence_scenario`; it is not a claim about a real connection or transfer airport.

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

## Second-level weather and news pages

The dashboard creates same-tab, shareable URLs for `GET /details/weather` and
`GET /details/news` after a successful comparison. Query parameters contain `origin`,
`destination`, `departure_time`, and `lang`. Before navigation, the dashboard stores the form,
active ranking, language, and latest comparison in `sessionStorage`; using Back restores that
session state when the browser still has it.

The weather page calls `POST /v1/context/weather-detail`. It shows both the origin at planned
departure and destination at route-model-estimated arrival, including current Open-Meteo model
conditions, the nearest target-hour forecast, a ±12-hour hourly window (up to 25 points), weather
risk components, provider validity/fallback metadata, and available NOAA METAR/TAF raw reports with
conservative bilingual explanations. Separate `aviation_metadata` explains an empty report list
(no ICAO, no applicable report, or provider failure). If an applicable NOAA report has the highest
risk, the overall risk source and validity timestamps follow that report. Automated report interpretation is informational and does not
replace an official aviation-weather briefing. Open-Meteo and aviation responses are fresh-cached
for 10 minutes; a successful weather payload up to six hours old may be displayed as a clearly
labelled stale cache if refresh fails. The page has manual refresh and auto-refreshes every 10
minutes.

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
`not_included`. A public airline student-program page is labelled `program_available`; it is not
labelled as an actual student fare for the requested trip. `confirmed_free` is reserved for
offer-level evidence, which the strict-free model-estimate mode normally does not have.

The student ranking follows the requested lexicographic order:

1. lower estimated price;
2. confirmed free checked baggage;
3. confirmed actual student discount;
4. confirmed free change/refund;
5. lower age and verification burden.

`program_available` alone does not satisfy the actual-discount criterion. Its published age and
verification metadata is considered only in the fifth criterion. Because estimated prices will
usually differ, later criteria most often act as tie-breakers.

## Source links and operational limits

- [NOAA Aviation Weather Data API](https://aviationweather.gov/data/api/)
- [Open-Meteo forecast API](https://open-meteo.com/en/docs)
- [Open-Meteo free API terms and attribution](https://open-meteo.com/en/terms)
- [FAA NAS Status](https://nasstatus.faa.gov/)
- [AirLabs schedules API](https://airlabs.co/docs/schedules)
- [ADSB.lol public API](https://www.adsb.lol/docs/open-data/api/)
- [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
- [GDELT Global Article List RSS](https://blog.gdeltproject.org/announcing-the-gdelt-article-list-rss-feed/)
- [GDELT project use and attribution](https://www.gdeltproject.org/about.html)
- [OurAirports public-domain data](https://ourairports.com/data/)

Free-provider quotas, fields, coverage, and terms can change. Review the linked sources before
deploying publicly or commercially. Use or redistribution of GDELT data must cite and link the
GDELT Project; the dashboard includes that visible attribution. API keys belong in the process
environment or another local secret store, never in the repository. The application does not
automatically load `.env`; set `AIRLABS_API_KEY` in the shell that starts the server.

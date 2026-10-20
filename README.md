# prophet-batch

Fit and forecast many [Prophet](https://github.com/facebook/prophet) models at once, on top of the stock
`prophet` package: batched exact MAP fits instead of one Stan run per series, and seeded, parallel forecasts
whose results depend only on (seed, series id).

Status: in development (not yet published). See the project repository for benchmarks and guarantees.

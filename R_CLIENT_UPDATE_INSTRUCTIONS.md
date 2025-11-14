# API Changes: Migration from Modality-Based to Model-ID-Based API

## Overview

This branch migrates the pysynthbio Python client from a modality-based API (`"bulk"`, `"single-cell"`) to a model-ID-based API (`"gem-1-bulk"`, `"gem-1-sc"`, etc.). The R client needs identical changes to maintain API compatibility.

---

## Summary of Changes

### 1. REMOVED Functions (Delete These)

- **`get_valid_modalities()`** - Previously returned `{"bulk", "single-cell"}`
- **`get_valid_query(modality)`** - Previously generated example queries based on modality string
- **`validate_query(query)`** - Client-side query validation (no longer needed)
- **`validate_modality(query)`** - Client-side modality validation (no longer needed)

### 2. ADDED Functions (Implement These)

#### `list_models(api_base_url = "https://app.synthesize.bio")`

**Purpose:** Fetch list of available models from the API

**Endpoint:** `GET /api/models`

**Authentication:** Requires Bearer token in Authorization header

**Returns:** JSON array of model objects with their IDs and metadata

**Python Implementation Reference:**

```python
def list_models(api_base_url: str = API_BASE_URL):
    if not has_synthesize_token():
        raise KeyError("No API token found...")

    url = f"{api_base_url}/api/models"
    response = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + os.environ["SYNTHESIZE_API_KEY"],
        },
    )
    models = response.json()
    return models
```

#### `get_example_query(model_id, api_base_url = "https://app.synthesize.bio")`

**Purpose:** Fetch model-specific example query from the API

**Parameters:**

- `model_id` (string, required): The model ID (e.g., `"gem-1-bulk"`, `"gem-1-sc"`)
- `api_base_url` (string, optional): API base URL

**Endpoint:** `GET /api/models/{model_id}/example-query`

**Authentication:** Requires Bearer token in Authorization header

**Returns:** JSON object with example query structure for the specified model

**Python Implementation Reference:**

```python
def get_example_query(model_id: str, api_base_url: str = API_BASE_URL):
    if not has_synthesize_token():
        raise KeyError("No API token found...")

    url = f"{api_base_url}/api/models/{model_id}/example-query"
    response = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + os.environ["SYNTHESIZE_API_KEY"],
        },
    )
    example_query = response.json()
    return example_query
```

### 3. MODIFIED Functions

#### `predict_query()` - BREAKING CHANGE

**New Required Parameter:** `model_id` (string)

**Before:**

```r
predict_query(query, as_counts = TRUE, ...)
```

**After:**

```r
predict_query(query, model_id, as_counts = TRUE, ...)
```

**Key Changes:**

1. Add `model_id` as a required parameter (should be second parameter after `query`)
2. Remove any internal modality detection/validation logic
3. Remove any calls to `validate_query()` or `validate_modality()`
4. Change API endpoint construction:
   - **OLD:** `/api/models/{api_slug}/predict` where api_slug was derived from modality
   - **NEW:** `/api/models/{model_id}/predict` where model_id is passed directly

**Implementation Notes:**

- The `query` object no longer needs validation on the client side
- The API handles all validation server-side
- Remove any hardcoded modality mappings (e.g., `"bulk"` → `"gem-1-bulk"`)

---

## API Endpoint Changes

### Old Endpoint Pattern

```
POST /api/models/{api_slug}/predict
```

Where `api_slug` was internally determined from `query["modality"]`:

- `"bulk"` → `"gem-1-bulk"`
- `"single-cell"` → `"gem-1-sc"`

### New Endpoint Pattern

```
POST /api/models/{model_id}/predict
```

Where `model_id` is explicitly provided by the user.

---

## Common Model IDs

Users should discover models via `list_models()`, but common examples include:

- `"gem-1-bulk"` - Bulk RNA-seq generation
- `"gem-1-sc"` - Single-cell RNA-seq generation (mean estimation only)
- `"gem-1-bulk_predict-metadata"` - Metadata prediction for bulk
- `"gem-1-sc_predict-metadata"` - Metadata prediction for single-cell

---

## Updated User Workflow

### OLD Workflow (Modality-Based)

```r
# 1. Get valid modalities
modalities <- get_valid_modalities()  # Returns {"bulk", "single-cell"}

# 2. Get example query for a modality
query <- get_valid_query(modality = "bulk")

# 3. Predict (modality inferred from query)
results <- predict_query(query = query, as_counts = TRUE)
```

### NEW Workflow (Model-ID-Based)

```r
# 1. List available models
models <- list_models()
print(models)

# 2. Get example query for a specific model
query <- get_example_query(model_id = "gem-1-bulk")

# 3. Predict (model_id explicitly provided)
results <- predict_query(
  query = query,
  model_id = "gem-1-bulk",
  as_counts = TRUE
)
```

---

## Code Examples for R Package

### Example: list_models()

```r
list_models <- function(api_base_url = "https://app.synthesize.bio") {
  if (!has_synthesize_token()) {
    stop("No API token found. Set SYNTHESIZE_API_KEY or call set_synthesize_token()")
  }

  url <- paste0(api_base_url, "/api/models")

  response <- httr::GET(
    url,
    httr::add_headers(
      Accept = "application/json",
      Authorization = paste("Bearer", Sys.getenv("SYNTHESIZE_API_KEY"))
    )
  )

  httr::stop_for_status(response)
  return(httr::content(response, as = "parsed"))
}
```

### Example: get_example_query()

```r
get_example_query <- function(model_id, api_base_url = "https://app.synthesize.bio") {
  if (!has_synthesize_token()) {
    stop("No API token found. Set SYNTHESIZE_API_KEY or call set_synthesize_token()")
  }

  url <- paste0(api_base_url, "/api/models/", model_id, "/example-query")

  response <- httr::GET(
    url,
    httr::add_headers(
      Accept = "application/json",
      Authorization = paste("Bearer", Sys.getenv("SYNTHESIZE_API_KEY"))
    )
  )

  httr::stop_for_status(response)
  return(httr::content(response, as = "parsed"))
}
```

### Example: Updated predict_query() signature

```r
predict_query <- function(query,
                         model_id,  # NEW REQUIRED PARAMETER
                         as_counts = TRUE,
                         auto_authenticate = TRUE,
                         api_base_url = "https://app.synthesize.bio",
                         poll_interval_seconds = 2,
                         poll_timeout_seconds = 900) {

  # Remove any validate_query() or validate_modality() calls

  # Add source field
  query$source <- "rsynthbio"  # or whatever your R package is called

  # Start model query with new endpoint
  url <- paste0(api_base_url, "/api/models/", model_id, "/predict")

  # Rest of implementation follows existing pattern...
}
```

---

## Documentation Updates Required

All documentation mentioning:

- `get_valid_modalities()` → Replace with `list_models()`
- `get_valid_query(modality="bulk")` → Replace with `get_example_query(model_id="gem-1-bulk")`
- `predict_query(query)` → Update to `predict_query(query, model_id="gem-1-bulk")`
- Modality strings (`"bulk"`, `"single-cell"`) → Replace with model IDs

### Key Documentation Sections to Update:

1. **Quickstart guide** - Update all code examples
2. **Query design guide** - Change from "Choosing a Modality" to "Choosing a Model"
3. **API reference** - Update function signatures and descriptions
4. **Examples/vignettes** - Update all working examples

---

## Testing Checklist

- [ ] `list_models()` successfully fetches model list from API
- [ ] `get_example_query("gem-1-bulk")` returns valid bulk query
- [ ] `get_example_query("gem-1-sc")` returns valid single-cell query
- [ ] `predict_query(query, model_id="gem-1-bulk")` generates bulk data
- [ ] `predict_query(query, model_id="gem-1-sc")` generates single-cell data
- [ ] All old function names (`get_valid_modalities`, `get_valid_query`) are removed
- [ ] Documentation updated and builds successfully
- [ ] All examples in documentation run without errors
- [ ] Package exports updated (NAMESPACE file)

---

## Breaking Changes Summary

**For End Users:**

1. `get_valid_modalities()` is removed - use `list_models()` instead
2. `get_valid_query(modality)` is removed - use `get_example_query(model_id)` instead
3. `predict_query()` now requires `model_id` parameter
4. Queries now use model IDs (e.g., `"gem-1-bulk"`) instead of modality strings (e.g., `"bulk"`)

**Migration Path:**

- `get_valid_modalities()` → `list_models()`
- `get_valid_query(modality="bulk")` → `get_example_query(model_id="gem-1-bulk")`
- `get_valid_query(modality="single-cell")` → `get_example_query(model_id="gem-1-sc")`
- `predict_query(query)` → `predict_query(query, model_id="gem-1-bulk")`

---

## Version Numbering Recommendation

This is a **major breaking change**. If your R package follows semantic versioning:

- If currently at version `2.x.x`, bump to `3.0.0`
- Clearly document breaking changes in CHANGELOG/NEWS

---

## Questions?

If the R package implementation differs significantly from the Python implementation described here, you may need to adapt these instructions. The core API changes remain the same regardless of client language.

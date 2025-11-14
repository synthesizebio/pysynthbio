# Instructions for AI Agent: Update R Client for New API

## Task

Update the R client package to match the API changes implemented in the pysynthbio Python package (branch: model_id\_\_example_query). The R package uses the same Synthesize Bio API.

## Core Changes Required

### 1. DELETE These Functions

- `get_valid_modalities()` - No longer needed
- `get_valid_query(modality)` - Replaced by `get_example_query(model_id)`
- `validate_query(query)` - Validation now server-side only
- `validate_modality(query)` - Validation now server-side only
- Any helper functions that map modality strings to API slugs

### 2. ADD These Two New Functions

**Function 1: `list_models()`**

```
Endpoint: GET {api_base_url}/api/models
Auth: Bearer token required
Returns: JSON array of available models
```

**Function 2: `get_example_query(model_id)`**

```
Endpoint: GET {api_base_url}/api/models/{model_id}/example-query
Auth: Bearer token required
Parameters: model_id (required string, e.g., "gem-1-bulk")
Returns: JSON object with example query structure for that model
```

### 3. MODIFY predict_query()

**Add required parameter:** `model_id` (string)

**Changes:**

- Add `model_id` as second parameter after `query`
- Change endpoint from `/api/models/{api_slug}/predict` to `/api/models/{model_id}/predict`
- Remove all internal modality-to-slug mapping logic
- Remove calls to `validate_query()` and `validate_modality()`
- Pass `model_id` directly to API endpoint

**New signature:**

```r
predict_query(query, model_id, as_counts = TRUE, ...)
```

## API Mapping Changes

**OLD (modality-based):**

- User specifies modality: `"bulk"` or `"single-cell"`
- Client maps to API slug: `"gem-1-bulk"` or `"gem-1-sc"`
- Endpoint: `/api/models/{api_slug}/predict`

**NEW (model-ID-based):**

- User specifies model_id directly: `"gem-1-bulk"`, `"gem-1-sc"`, etc.
- No client-side mapping needed
- Endpoint: `/api/models/{model_id}/predict`

## Documentation Updates

Replace all instances:

- `get_valid_modalities()` → `list_models()`
- `get_valid_query(modality = "bulk")` → `get_example_query(model_id = "gem-1-bulk")`
- `predict_query(query)` → `predict_query(query, model_id = "gem-1-bulk")`

Update sections:

- Quickstart guide: Update all code examples to use new functions
- Query design guide: Change "Choosing a Modality" to "Choosing a Model"
- Function documentation: Update signatures and parameter descriptions

## Example User Workflow Change

**BEFORE:**

```r
modalities <- get_valid_modalities()
query <- get_valid_query(modality = "bulk")
results <- predict_query(query)
```

**AFTER:**

```r
models <- list_models()
query <- get_example_query(model_id = "gem-1-bulk")
results <- predict_query(query, model_id = "gem-1-bulk")
```

## Reference Implementation

See Python implementation in:

- `/Users/jmkahn/Projects/pysynthbio/src/pysynthbio/list_models.py`
- `/Users/jmkahn/Projects/pysynthbio/src/pysynthbio/get_example_query.py`
- `/Users/jmkahn/Projects/pysynthbio/src/pysynthbio/call_model_api.py` (predict_query)

## Success Criteria

- [ ] Old functions removed
- [ ] New functions implemented and working
- [ ] `predict_query()` requires `model_id` parameter
- [ ] All documentation updated
- [ ] All examples run successfully
- [ ] Package exports updated (NAMESPACE)

This is a breaking change - consider bumping major version (e.g., 2.x → 3.0).

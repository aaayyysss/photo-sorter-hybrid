# API Schema (v0)

## POST /api/refs/register
Register reference persons with a set of embeddings.

**Request JSON**
```json
{
  "persons": [
    {
      "name": "Alice",
      "embeddings": [[0.12, -0.33, ...], [0.11, -0.31, ...]]
    },
    {
      "name": "Bob",
      "embeddings": [[...], [...]]
    }
  ],
  "normalize": true
}
```

**Response**
```json
{ "status": "ok", "registered": ["Alice", "Bob"] }
```

## POST /api/sort
Ask the server to assign inbox embeddings to the closest person.

**Request JSON**
```json
{
  "inbox": [
    { "file": "IMG_0001.jpg", "embedding": [ ... ] },
    { "file": "IMG_0002.jpg", "embedding": [ ... ] }
  ],
  "threshold": 0.32,
  "multi_label": false
}
```

**Response**
```json
{
  "status": "ok",
  "assignments": [
    { "file": "IMG_0001.jpg", "best": {"person": "Alice", "score": 0.71}, "all": [{"person": "Alice","score":0.71}, {"person":"Bob","score":0.22}] },
    { "file": "IMG_0002.jpg", "best": null, "all": [{"person":"Alice","score":0.29},{"person":"Bob","score":0.10}] }
  ]
}
```

- `score` is cosine similarity in [0..1].
- If `best` is null, no one crossed the threshold.

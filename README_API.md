# Python CNN API

Endpoint:

- `GET /` — info service
- `GET /health` — health check
- `POST /predict` — multipart form-data, field `image`

Contoh JSON berhasil:

```json
{
  "ok": true,
  "label": "BROKOLI",
  "class_index": 0,
  "confidence": 87.42,
  "raw_score": 0.1258,
  "threshold": 0.5,
  "input_size": [150, 150, 3]
}
```

Model dimuat dari `model/model_cnn_brokoli_kembangkol.keras`.

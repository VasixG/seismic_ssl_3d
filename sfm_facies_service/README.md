# SFM Facies Service (FastAPI + Streamlit)

Локальный сервис для кластеризации фасций/геометрий по xarray NetCDF кубам.

## Что умеет
- Загрузка `.nc` xarray куба
- Срезы: time slice, window aggregation, inline, xline
- Нормализация, тайлинг 224/512, ViT patch features
- Кластеризация (KMeans), embedding (PCA)
- Визуализация: срез, кластеры, оверлей, embedding
- 3D кластеризация по всему кубу (time-slices) + 3D просмотр и сечения

## Структура
```
backend/
  app.py
  sfm.py
  slicers.py
  clustering.py
  store.py
  requirements_backend.txt
frontend/
  ui.py
  requirements_frontend.txt
_runs/           # локальные результаты
weights/         # сюда положи чекпоинты (не в git)
```

## Docker
1) Положи веса SFM в `sfm_facies_service/weights/` по подпапкам model_size/tile_size.
2) Запуск:
```
docker compose up --build
```
3) Открой:
- Backend: http://localhost:8000
- Frontend (UI): http://localhost:8501

Сервис сам подхватывает веса по `model_size` и `tile_size`.
Разложи `.pth` так:
```
weights/
  base/
    224/
      Base-224.pth
    512/
      Base-512.pth
  large/
    224/
      Large-224.pth
    512/
      Large-512.pth
```
Для Docker по умолчанию используется `sfm_facies_service/weights` (монтируется в `/weights`).

## Dev режим (без переустановки зависимостей при изменении кода)
Код монтируется в контейнеры, зависимости остаются в образе.
```
docker compose -f docker-compose.dev.yml up --build
```

## Локальный запуск без Docker
Backend:
```
cd backend
pip install -r requirements_backend.txt
export SFM_SERVICE_DATA=./_runs
uvicorn app:app --host 0.0.0.0 --port 8000
```

Frontend:
```
cd frontend
pip install -r requirements_frontend.txt
export API_URL=http://localhost:8000
streamlit run ui.py --server.maxUploadSize 2048
```

## Примечания
- Ожидается одна `data_var` и dims `(iline, xline, twt)`.
- Если CUDA недоступна, backend автоматически переключится на CPU.

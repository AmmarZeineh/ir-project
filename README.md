# نظام استرجاع المعلومات (Information Retrieval System)

مشروع استرجاع معلومات مبني على **معمارية الخدمات (SOA)**، يدعم المعالجة المسبقة،
الفهرسة، البحث اللفظي والدلالي، البحث الهجين، تحسين الاستعلام، التقييم،
و**RAG** كطلب إضافي.

---

## مجموعات البيانات

المشروع يستخدم datasetين من [ir-datasets](https://ir-datasets.com/)، كل واحد فيه
أكثر من 200K وثيقة وملف `qrels` رسمي:

| Dataset | المصدر | الوثائق | الاستعلامات |
|---------|--------|---------|-------------|
| **Quora** | `beir/quora/test` | 522,931 | 10,000 |
| **Touche-2020** | `beir/webis-touche2020/v2` | 382,545 | 50 |

> **ملاحظة للتسليم:** dataset واحد يكفي حسب متطلب المشروع. عندنا الاثنين مطبّقين
> عليهم نفس المتطلبات.

---

## هيكل المشروع

```
Ir-project/
├── shared/
│   ├── docstore.py          # SQLite — تخزين النص الأصلي (raw) لكل وثيقة
│   ├── data_paths.py        # مسارات data/
│   └── schemas.py           # نماذج الطلبات والاستجابات
├── services/
│   ├── preprocessing_service/    # :8001
│   ├── indexing_service/         # :8002
│   ├── retrieval_service/        # :8003  (TF-IDF, BM25, Embedding, Hybrid, RAG)
│   ├── ranking_service/          # :8004
│   ├── query_refinement_service/ # :8005
│   └── api_gateway/                # :8000  (الواجهة + /search + /rag)
├── notebooks/               # سكربتات التحميل، المعالجة، التقييم، والرسوم
├── data/                    # غير مرفوع على Git (انظروا قسم التسليم أدناه)
│   ├── raw/<dataset>/
│   ├── processed/<dataset>/
│   ├── indexes/<dataset>/
│   └── docstore/<dataset>.db
├── start_services.sh
└── requirements.txt
```

---

## المعمارية (SOA)

```
┌─────────────────────────────────────────────────────────────┐
│                    Web UI  (:8000)                          │
│              api_gateway  —  /search  /rag                  │
└──────────┬──────────────┬──────────────┬────────────────────┘
           │              │              │
           ▼              ▼              ▼
   ┌──────────────┐ ┌─────────────┐ ┌──────────────────┐
   │ Preprocess   │ │  Retrieval  │ │ Query Refinement │
   │   :8001      │ │    :8003    │ │      :8005       │
   └──────────────┘ └──────┬──────┘ └──────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        Indexes       Embeddings    SQLite Docstore
     (pickle/csv)      (.npy)      (raw text by doc_id)
```

### الخدمات

| الخدمة | المنفذ | الوظيفة |
|--------|--------|---------|
| API Gateway | 8000 | الواجهة، تجميع الطلبات، قراءة النص الأصلي من DB |
| Preprocessing | 8001 | تطبيع، tokenization، lemmatization |
| Indexing | 8002 | بناء inverted index + IDF |
| Retrieval | 8003 | TF-IDF, BM25, Embedding, Hybrid, RAG |
| Ranking | 8004 | حساب MAP, P@10, Recall, nDCG |
| Query Refinement | 8005 | تصحيح إملائي + توسيع مرادفات |

### تدفق البحث (مهم للتقييم)

1. المستخدم يرسل استعلاماً عبر الواجهة.
2. (اختياري) Query Refinement يحسّن الاستعلام.
3. Preprocessing يعالج الاستعلام بنفس خطوات الوثائق.
4. Retrieval يبحث على **النص المعالج** ويرجع `doc_id` + `score` فقط.
5. API Gateway يقرأ **النص الأصلي (raw)** من SQLite حسب `doc_id`.
6. الواجهة تعرض أعلى 10 وثائق بالنص الأصلي، مو المنضف.

---

## التثبيت

```shell
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet
```

لتوليد رسوم التقييم:

```shell
python -m pip install matplotlib
```

---

## تشغيل المشروع

#### Quora

```shell
python notebooks/01_load_quora.py      # تنزيل البيانات
python notebooks/03_pipeline_quora.py    # معالجة + فهرسة + embeddings
python notebooks/05_build_docstore.py    # بناء SQLite للنص الأصلي
python notebooks/06_evaluate_quora.py    # تقييم
```

#### Touche-2020

```shell
python notebooks/02_load_touche.py
python notebooks/04_pipeline_touche.py
python notebooks/05_build_docstore.py
python notebooks/07_evaluate_touche.py
```

#### تشغيل الخدمات

```shell
chmod +x start_services.sh
./start_services.sh
```

---

## نماذج الاسترجاع

| النموذج | الوصف |
|---------|-------|
| **TF-IDF** | VSM مع log-TF × IDF وتطبيع طول الوثيقة |
| **BM25** | نموذج احتمالي مع `k1` و `b` قابلين للتعديل من الواجهة |
| **Embedding** | `all-MiniLM-L6-v2` + cosine similarity |
| **Hybrid Serial** | BM25 لاختيار مرشحين → إعادة ترتيب بالـ embedding |
| **Hybrid Parallel (RRF)** | دمج BM25 + TF-IDF + Embedding بـ Reciprocal Rank Fusion |
| **Hybrid Parallel (Linear)** | دمج خطي مع تطبيع الدرجات |

---

## الطلب الإضافي: RAG

- تبويب **RAG Chat** بالواجهة — قابل للتجربة **بشكل مستقل** عن البحث العادي.
- يسترجع وثائق بـ BM25 + Embedding، يقرأ النص الأصلي من SQLite، ثم يولّد جواباً.
- **Ollama** (اختياري للتوليد الكامل):

```shell
```

إذا Ollama مو شغال، النظام يستخدم fallback استخراجي (extractive).

---

## التقييم (Evaluation)

المقاييس: **MAP**, **P@10**, **Recall**, **nDCG@10**

```shell
python notebooks/06_evaluate_quora.py    # عينة 200 استعلام (random_state=42)
python notebooks/07_evaluate_touche.py   # كل الاستعلامات اللي عندها qrels
python notebooks/08_plot_results.py      # رسوم بيانية PNG
```

النتائج:

- `data/processed/quora/evaluation_results.csv`
- `data/processed/touche/evaluation_results.csv`
- `data/processed/*/plots/*.png`

---

## تجربة الخدمات بشكل مستقل

```shell
# صحة الخدمات
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8003/health

# معالجة نص
curl -X POST http://localhost:8001/preprocess \
  -H "Content-Type: application/json" \
  -d '{"text": "how to invest in stocks"}'

# بحث
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how to invest", "dataset": "quora", "model": "bm25", "top_k": 10}'

# RAG
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "how to invest in share market", "dataset": "quora", "top_k": 5}'
```

---

## التسليم — ملفات البيانات (data/)

مجلد `data/` **غير مرفوع على GitHub** (موجود بـ `.gitignore`) لأن حجمه كبير
(عدة GB).

## المتطلبات المحققة

| # | المتطلب | الحالة |
|---|---------|--------|
| 1 | معالجة البيانات | ✅ |
| 2 | TF-IDF, BM25, Embedding, Hybrid Serial/Parallel | ✅ |
| 3 | الفهرسة (Inverted Index) | ✅ |
| 4 | معالجة الاستعلام | ✅ |
| 5 | Query Refinement | ✅ |
| 6 | المطابقة والترتيب | ✅ |
| 7 | SOA | ✅ |
| 8 | التقييم (MAP, nDCG, ...) | ✅ |
| 9 | واجهة مستخدم | ✅ |
| + | RAG (طلب إضافي — مجموعة 5) | ✅ |
| + | raw text من DB وقت الـ query | ✅ |

---

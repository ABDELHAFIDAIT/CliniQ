# CliniQ
Assistant intelligent basé sur une architecture RAG optimisée, fournissant aux professionnels de santé un accès instantané et contextualisé aux protocoles médicaux et à la documentation clinique.

```
├── 📁 alembic
│   ├── 📁 versions
│   └── 🐍 env.py
├── 📁 app
│   ├── 📁 api
│   │   ├── 📁 endpoints
│   │   ├── 📁 middlewares
│   │   │   └── 🐍 log_middleware.py
│   │   └── 🐍 router.py
│   ├── 📁 core
│   │   ├── 🐍 config.py
│   │   ├── 🐍 exceptions.py
│   │   ├── 🐍 logging.py
│   │   └── 🐍 security.py
│   ├── 📁 db
│   │   ├── 🐍 base.py
│   │   └── 🐍 session.py
│   ├── 📁 models
│   │   ├── 🐍 query.py
│   │   └── 🐍 user.py
│   ├── 📁 schemas
│   │   └── 🐍 user.py
│   ├── 📁 services
│   │   ├── 🐍 eval_service.py
│   │   ├── 🐍 rag_service.py
│   │   └── 🐍 vector_store.py
│   ├── 🐍 exceptions_handler.py
│   └── 🐍 main.py
├── 📁 frontend
│   ├── 📁 pages
│   ├── 📁 utils
│   ├── 🐳 Dockerfile
│   └── 🐍 main.py
├── 📁 monitoring
│   ├── 📁 grafana
│   └── 📁 prometheus
├── 📁 scripts
│   └── 🐍 ingest_doc.py
├── 📁 tests
│   ├── 🐍 test_api.py
│   └── 🐍 test_rag.py
├── ⚙️ .env_example
├── ⚙️ .gitignore
├── 🐳 Dockerfile
├── 📝 README.md
├── ⚙️ alembic.ini
├── ⚙️ docker-compose.yaml
└── 📄 requirements.txt
```
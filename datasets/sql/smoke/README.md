# SQL Smoke Dataset

The first SQL smoke set uses a tiny generated SQLite fixture named `company_small`.

The fixture is built by code, not checked in as a binary database.

Tables:

- `departments(id, name)`
- `employees(id, name, department_id, salary)`

The smoke eval compares predicted SQL and gold SQL by result equivalence.

Current defaults:

- `order_sensitive = false`
- `numeric_tolerance = 1e-6`
- raw SQL only


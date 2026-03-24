# Optuna Optimization with PostgreSQL Workflow

This guide provides a comprehensive workflow for performing distributed and parallel model optimization using Optuna with a PostgreSQL backend.

## 1. PostgreSQL Server Setup (Host PC)

Choose one PC to host the database. The other PC will connect to it over the network.

### Installation
```bash
sudo apt update && sudo apt install postgresql postgresql-contrib
```

### Create User and Database
```bash
# Log into PostgreSQL prompt
sudo -u postgres psql

# Run these commands in the psql prompt:
CREATE USER optuna_user WITH PASSWORD 'password';
CREATE DATABASE optuna_db OWNER optuna_user;
\q
```

### Network Configuration (For Multi-PC setups)
1. **Enable Listening:** Edit `/etc/postgresql/16/main/postgresql.conf` (version may vary).
   Change `#listen_addresses = 'localhost'` to:
   ```text
   listen_addresses = '*'
   ```
2. **Configure Authentication:** Edit `/etc/postgresql/16/main/pg_hba.conf`.
   Add this line at the end to allow your local network:
   ```text
   host    optuna_db    optuna_user    192.168.5.0/24    md5
   ```
3. **Restart PostgreSQL:**
   ```bash
   sudo systemctl restart postgresql
   ```
4. **Open Firewall:**
   ```bash
   sudo ufw allow 5432/tcp
   ```

---

## 2. Python Environment Setup (All PCs)

Ensure each PC has the necessary Python packages installed:

```bash
pip install optuna optuna-dashboard psycopg2-binary
```

---

## 3. Running Optimization

Your Host PC (PC-A) IP is **192.168.5.77**.

### On the Host PC (PC-A: 192.168.5.77)
```bash
python apps/optimize_model.py <other_args> \
  --storage postgresql://optuna_user:password@localhost/optuna_db \
  --study-name mlp_optimization \
  --n-jobs 4
```

### On the Remote Worker (PC-B: 192.168.5.65)
```bash
python apps/optimize_model.py <other_args> \
  --storage postgresql://optuna_user:password@192.168.5.77/optuna_db \
  --study-name mlp_optimization \
  --n-jobs 4
```

*   **`--storage`**: The connection string to the PostgreSQL database.
*   **`--study-name`**: Must be the same on both PCs to share the same study.
*   **`--n-jobs`**: Number of parallel jobs per PC.

---

## 4. Resuming or Extending a Study

One of the main benefits of using an RDB backend is the ability to resume or extend an optimization session.

*   **To Resume:** If a script crashes or you manually stop it, just run the same command again. Optuna will automatically load the existing trials and continue.
*   **To Extend:** If you completed 100 trials and want 200 more, simply run the command again with `--trials 200`. Optuna will add 200 **new** trials to the database.

---

## 5. Visualization and Analysis

### Using Optuna Dashboard (Interactive)
Run this on either PC to monitor the progress in real-time via a web browser:
```bash
optuna-dashboard postgresql://optuna_user:password@192.168.5.77/optuna_db
```
Then visit `http://localhost:8080`.

### Custom Analysis with Python
You can load the data into a Pandas DataFrame for custom plotting:
```python
import optuna

study = optuna.load_study(
    study_name="mlp_optimization",
    storage="postgresql://optuna_user:password@localhost/optuna_db"
)

# Export to CSV
df = study.trials_dataframe()
df.to_csv("optimization_results.csv")
```

---

## 6. Maintenance

### Deleting a Study
If you want to start fresh without deleting the whole database:
```bash
optuna delete-study --study-name mlp_optimization --storage postgresql://optuna_user:password@localhost/optuna_db
```

### Backing up the Database
```bash
pg_dump -U optuna_user optuna_db > optuna_backup.sql
```

# VisionAI
Here's a step-by-step guide to set up your environment on **Windows**:  

---

### **1. Install Prerequisites**  

#### **1.1 Install Docker Desktop**  
1. Download and install **Docker Desktop** from [Docker's official site](https://www.docker.com/products/docker-desktop/).  
2. Enable **WSL 2 backend** (recommended) or Hyper-V during installation.  
3. Ensure Docker is running by opening **PowerShell** and running:  
   ```powershell
   docker --version
   ```
4. Enable Docker Compose:  
   ```powershell
   docker-compose --version
   ```

---

### **2. Install `pipx` and Poetry**  

#### **2.1 Install `pipx` (if not installed)**  
```powershell
python -m pip install --user pipx
python -m pipx ensurepath
```
**Restart PowerShell or Command Prompt for changes to take effect.**  

#### **2.2 Install Poetry via `pipx`**  
```powershell
pipx install poetry
```
Check installation:  
```powershell
poetry --version
```

---

### **3. Install Dependencies**  
Navigate to your project folder:  
```powershell
cd path\to\your\project
```
Install dependencies **without installing the project itself**:  
```powershell
poetry install --no-root
```

---

### **4. Start Docker Compose**  
Ensure you're in the project directory with the `docker-compose.yml` file, then run:  
```powershell
docker-compose up -d
```
This will start your Docker containers in detached mode (`-d`).

---

### **5. Start the Uvicorn Server Using Poetry**  
If using `poetry run`:  
```powershell
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### **6. Verify Everything is Running**  
- Open your FastAPI app in a browser:  
  ```
  http://localhost:8000
  ```

---

Let me know if you need additional setup details! 🚀
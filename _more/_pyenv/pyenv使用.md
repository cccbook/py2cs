你可以依照你的作業系統，選擇適合的方法將 Python 降級到 3.10。

---

## **Windows**
### **方法 1：使用 `pyenv`（推薦）**
如果你已安裝 **pyenv**，可以透過它管理 Python 版本：
1. 安裝 **pyenv**（如果尚未安裝）：
   ```powershell
   iwr -useb https://pyenv.run | iex
   ```
2. 重新啟動 PowerShell，然後安裝 Python 3.10：
   ```powershell
   pyenv install 3.10.13
   ```
3. 設定全域 Python 版本：
   ```powershell
   pyenv global 3.10.13
   ```
4. 確認 Python 版本：
   ```powershell
   python --version
   ```

### **方法 2：手動下載安裝**
1. 先移除現有 Python：
   - **Windows 設定** → **應用程式** → **已安裝應用程式** → **Python** → 移除
2. 下載 **Python 3.10**：
   - 前往 [Python 官網](https://www.python.org/downloads/release/python-31013/)
   - 下載適用於 Windows 的 **installer**
3. 安裝時 **勾選 "Add Python to PATH"**，然後完成安裝。
4. 確認安裝成功：
   ```powershell
   python --version
   ```

---

## **Linux (Ubuntu / Debian)**
### **方法 1：使用 `pyenv`（推薦）**
1. 安裝 `pyenv`（如果尚未安裝）：
   ```bash
   curl https://pyenv.run | bash
   ```
   然後在 `~/.bashrc`（或 `~/.bash_profile`）中加入：
   ```bash
   export PATH="$HOME/.pyenv/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv virtualenv-init -)"
   ```
   重新載入設定：
   ```bash
   source ~/.bashrc
   ```
2. 安裝 Python 3.10：
   ```bash
   pyenv install 3.10.13
   pyenv global 3.10.13
   ```
3. 確認版本：
   ```bash
   python --version
   ```

### **方法 2：手動安裝**
1. 先移除現有 Python：
   ```bash
   sudo apt remove python3
   ```
2. 安裝 Python 3.10：
   ```bash
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev -y
   ```
3. 設定 `python` 指向 `python3.10`：
   ```bash
   sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
   ```
4. 確認版本：
   ```bash
   python --version
   ```

---

## **macOS**
### **方法 1：使用 `pyenv`（推薦）**
1. 安裝 `pyenv`：
   ```bash
   brew install pyenv
   ```
2. 安裝 Python 3.10：
   ```bash
   pyenv install 3.10.13
   pyenv global 3.10.13
   ```
3. 確認版本：
   ```bash
   python --version
   ```

### **方法 2：使用 Homebrew**
1. 先移除舊版 Python：
   ```bash
   brew uninstall python
   ```
2. 安裝 Python 3.10：
   ```bash
   brew install python@3.10
   ```
3. 設定 `python` 指向 `python3.10`：
   ```bash
   echo 'export PATH="/usr/local/opt/python@3.10/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```
4. 確認版本：
   ```bash
   python --version
   ```

---

試試看這些方法，確保 `python --version` 正確顯示 **3.10.x**！
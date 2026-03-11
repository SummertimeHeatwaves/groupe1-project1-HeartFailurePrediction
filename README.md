# groupe1-project1-HeartFailurePrediction
# Git, Testing, and Continuous Integration Lab

This repository is used in the **Collaborative Programming lab** to introduce several important software engineering practices used in real-world development:

* **Git** for version control
* **GitHub** for collaboration
* **pytest** for automated testing
* **GitHub Actions** for Continuous Integration (CI)

You will complete several Python exercises and submit your work using a **Pull Request workflow**.

---

## Repository Structure

```
learn-git-pytest
│
├── exercise1/
│   └── calculator.py
│
├── exercise2/
│   └── strings.py
│
├── exercise3/
│   └── temperature.py
│
├── tests/
│   ├── test_exercise1.py
│   ├── test_exercise2.py
│   └── test_exercise3.py
│
└── requirements.txt
```

Each exercise contains incomplete Python functions that you must implement.

The `tests/` folder contains automated tests that verify your solutions.

---

## Learning Objectives

By completing this lab you will learn:

#### Version Control (Git)

* Forking a repository
* Cloning a repository
* Creating branches
* Committing changes
* Pushing code to GitHub

#### Automated Testing

* Running tests with **pytest**
* Understanding test failures
* Fixing code until tests pass

#### Continuous Integration

* Running automated checks using **GitHub Actions**
* Understanding CI pipelines
* Verifying code quality automatically

---
You can write the section like this so it stays **simple and consistent** for students.

---

## Required Software

You must install the following tools.

---

### Install Using `winget` (recommended for Windows)

If you are using **Windows**, you can install some tools directly from the terminal.

Open **PowerShell** and run:

#### Install Git (If not already installed)

```powershell
winget install Git.Git
```

Verify installation:

```bash
git --version
```

---

#### Install Python (If not already installed)

```powershell
winget search Python.Python.3.11
```

Verify installation:

```bash
python --version
```

---

#### Install VS Code (recommended)

```powershell
winget install Microsoft.VisualStudioCode
```

After installation, open VS Code and install the **Python extension**.

---

### Install GitHub Desktop

Download and install from the official website:

[https://github.com/apps/desktop](https://github.com/apps/desktop)

After installing, log in with your **GitHub account**.

GitHub Desktop provides a **graphical interface for Git**, which makes it easier to manage repositories, commits, and branches.

---


## Step 1 — Fork the Repository

Go to the course repository on GitHub.

Click the **Fork** button in the top-right corner.

This creates your **own copy of the repository** under your GitHub account.

---

## Step 2 — Clone Your Fork

Clone your fork to your computer.

#### Using GitHub Desktop

1. Open GitHub Desktop
2. Click **File → Clone repository**
3. Select your fork
4. Choose a location on your computer

---

## Step 3 — Open the Project

Open the project folder using **VS Code**.

---

## Step 4 — Create a Python Virtual Environment

In the project folder, open a terminal and run:

```bash
python -m venv .venv
```

This creates an isolated Python environment.

---

### Activate the Environment

#### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

Si il y a un problème
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Windows (Command Prompt)

```cmd
.venv\Scripts\activate
```

You should now see:

```
(.venv)
```

in your terminal prompt.

---



## Step 5 — Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

This installs:

* pytest
* mypy
* other required tools

---

## Step 6 — Run the Tests

Run the test suite:

```bash
pytest tests/
```

You will notice that **some tests fail**.

Your task is to **modify the code so that all tests pass**.

---

## Step 7 — Create Your Working Branch

Before modifying the code, create a branch with your name.

Example:

```bash
git checkout -b firstname-lastname
```

Replace `firstname-lastname` with your name.

Example:

```
git checkout -b alice-smith
```

---

## Step 8 — Implement the Functions

Complete the missing functions in the exercise files.

Example files:

```
exercise1/calculator.py
exercise2/strings.py
exercise3/temperature.py
```

Run tests frequently:

```bash
pytest tests/
```

When all tests pass, your implementation is correct.

---

## Step 9 — Commit Your Changes

Save your progress using Git.

```bash
git add .
git commit -m "Implement solutions for exercises"
```

A **commit** records a snapshot of your work.

---

## Step 10 — Push Your Branch

Upload your branch to your GitHub fork:

```bash
git push -u origin firstname-lastname
```

---

## Continuous Integration (GitHub Actions)

This repository uses **GitHub Actions** to automatically verify your code.

Each time you push code:

1. GitHub downloads your repository
2. Installs Python
3. Installs dependencies
4. Runs type checking with **mypy**
5. Runs all tests using **pytest**

You can see the results in the **Actions** tab of your GitHub repository.

If tests fail, you must fix your code and push again.

---

## Step 11 — Submit Your Work

When all tests pass locally and in GitHub Actions:

1. Go to the **original course repository**
2. Click **Pull Requests**
3. Click **New Pull Request**
4. Click **compare across forks**

Then configure:

* **Base repository:** course repository
* **Base branch:** `main`
* **Head repository:** your fork
* **Compare branch:** `firstname-lastname`

---

## Step 12 — Create the Pull Request

Add a title such as:

```
Exercise Solutions - Firstname Lastname
```

Click **Create Pull Request**.

---

## What Happens Next

Your instructor will:

* Review your code
* Verify that all tests pass
* Provide feedback in the Pull Request
* Approve or request changes

---

## Summary of the Workflow

```
Fork repository
Clone repository
Create branch
Write code
Run tests locally
Commit changes
Push branch
Create Pull Request
Check GitHub Actions
```

---

Happy coding! 🚀

# Document Embeddings with InterSystems IRIS Cloud SQL

This project demonstrates how to create embeddings from documents using **Nomic’s multimodal embedding models** and store them in **InterSystems IRIS Cloud SQL**. It also provides a **Streamlit UI** to search across your embedded data.

---
<p align="center">
<img src="image.png" alt="screenshot" style="max-width:700px; max-height:1000px;">
</p>

## 0. Requirements

This project requires:

* **Python 3.11+**
* **pip** or [**uv**](https://docs.astral.sh/uv/) for environment setup
* **GPU** (for local inference with `nomic-ai/colnomic-embed-multimodal-3b` or `7b`)

  * Recommended at least 12gb VRAM: **NVIDIA T4 or RTX 3090**

---

## 1. Cloud SQL Trial

To get started, you’ll need access to the **InterSystems Cloud SQL Trial**:

1. Sign up at [InterSystems Cloud SQL Trial](https://cloud.intersystems.com).
2. Create a new **SQL database** instance.
3. Copy your **connection details** (host, port, user, password, namespace).
4. Download the connection certificate from cloud SQL and rename connection.pem (and put in the connection folder.)

Here’s a step-by-step guide to getting set up with **InterSystems IRIS Cloud SQL**, whether you're using the free trial or a paid subscription:

### Step 1: Access the Cloud Services Portal

Head over to the **InterSystems Cloud Services Portal** to begin — here you can sign up for either a **Cloud SQL trial** or a **paid subscription**, and create a deployment.([InterSystems Documentation][1])

### Step 2: Navigate to the Services Page

Once inside the portal, go to the **Services** section (usually under the main menu). There, you'll find a card for **InterSystems IRIS Cloud SQL** where you can click **Subscribe** to view pricing and subscription options.([InterSystems Documentation][2])

### Step 3: Choose Trial or Subscription

* **Trial access**: Often accompanied by introductory credits (e.g., the site mentions “\$300 in FREE credits”).([InterSystems Corporation][3])
* **Paid Subscription**: If you're ready for production-grade use, the portal lets you subscribe directly or via your cloud provider (e.g., AWS Marketplace). Billing is handled through InterSystems or your cloud account, depending on your selection.([InterSystems Documentation][2])

### Step 4: Launch a New Deployment

After subscribing, head to the **Deployments** page. Click **Create New Deployment** and follow the prompts to configure details such as size, region, service level (e.g., Development, Test, Live), and any additional options like external connectivity.([InterSystems Documentation][2])

### Step 5: Manage and Configure Your Deployment

Once created, you can access your deployment’s **Overview** page to:

* View connection details (host, port, namespace, etc.)
* Enable or disable external access as needed (`0.0.0.0/0` or other subnet)
* Adjust security options like IP restrictions or TLS settings([InterSystems Documentation][1], [InterSystems Documentation][4])
* Download the SQL pem certificate and rename it connection.pem

---

### Optional: Video Walkthrough

If you'd prefer a quick visual tour, there's a helpful video demonstration titled *“Working with InterSystems IRIS Cloud SQL”* that walks through deployment and connection processes.([InterSystems Developer Community][6])

---

### Summary Table

| Step | Action                                                                                |
| ---- | ------------------------------------------------------------------------------------- |
| 1️⃣  | Log in to **InterSystems Cloud Services Portal**                                      |
| 2️⃣  | Navigate to **Services** → choose **Cloud SQL** → **Subscribe**                       |
| 3️⃣  | Select **Trial** (if available) or **Paid Subscription**                              |
| 4️⃣  | Under **Deployments**, click **Create New Deployment**                                |
| 5️⃣  | Configure your deployment and then access its **Overview**                            |
| 6️⃣  | Download driver → gather connection info → connect securely (e.g., via Python DB-API) |

---

[1]: https://docs.intersystems.com/services/csp/docbook/DocBook.UI.Page.cls?KEY=PAGE_iriscloudsql&utm_source=chatgpt.com "Welcome to InterSystems IRIS Cloud SQL"
[2]: https://docs.intersystems.com/services/csp/docbook/DocBook.UI.Page.cls?KEY=ISCSP_reference&utm_source=chatgpt.com "Cloud Services Portal Reference Information | InterSystems Cloud ..."
[3]: https://www.intersystems.com/products/intersystems-iris-cloud-services/cloud-sql/?utm_source=chatgpt.com "InterSystems IRIS Cloud SQL"
[4]: https://docs.intersystems.com/components/csp/docbook/DocBook.UI.Page.cls?KEY=GDRIVE_cloudsql&utm_source=chatgpt.com "Connecting Your Application to InterSystems IRIS Cloud SQL"
[5]: https://community.intersystems.com/post/cloud-sql?utm_source=chatgpt.com "Cloud SQL - InterSystems Developer Community"
[6]: https://community.intersystems.com/post/video-working-intersystems-iris-cloud-sql?utm_source=chatgpt.com "[Video] Working with InterSystems IRIS Cloud SQL"


## 2. Setting up Project

Clone the repo and install dependencies:

```bash
git clone https://github.com/isc-nmitchko/iris-document-search.git
cd iris-document-search
pip -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run setup to configure your environment:

```bash
python main.py --setup
```

This will create a `.env` file containing your IRIS connection string and model settings.

---

## 4. Embedding Data

You can embed documents (PDFs, text files, etc.) using the **Nomic multimodal embedding models**:

* `"nomic-ai/colnomic-embed-multimodal-3b"` (lighter, faster)
* `"nomic-ai/colnomic-embed-multimodal-7b"` (larger, more accurate)

Example usage:

```bash
python main.py --embed ./data/my_docs/
```

This will:

1. Extract text from your documents.
2. Generate embeddings using the configured model.
3. Insert the embeddings into your **InterSystems IRIS Cloud SQL** instance.

---

## 5. Searching using Streamlit UI

To search across embedded data, launch the Streamlit UI:

```bash
streamlit run ui2.py
```

Features include:

* Free-text semantic search
* Similarity ranking of documents
* Viewing results directly in a web interface
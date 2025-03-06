<div align="center">
  <h1>Phi3-rust 🦀🚀</h1>
  <p>Serve Phi3 with Candle and Actix for blazing-fast inference.</p>

  <a href="https://github.com/d1pankarmedhi/Phi3-rust/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
  <img src="https://img.shields.io/badge/Rust-1.70+-orange.svg" alt="Rust Version">
  <img src="https://img.shields.io/badge/Candle-%F0%9F%97%A8-blue.svg" alt="Candle">
  <img src="https://img.shields.io/badge/Actix--web-%F0%9F%9B%A8-green.svg" alt="Actix-web">
</div>

---

## 🚀 Overview

Phi3-rust provides a high-performance RESTful API for serving the [Phi3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) model. 🤖 Leveraging the power of [Candle](https://github.com/huggingface/candle/tree/main) for efficient inference and [Actix-web](https://github.com/actix/actix-web) for robust web serving, this project delivers lightning-fast responses. ⚡

## 🛠️ Key Components

* **Candle:** 🕯️ For optimized and efficient Phi3 model inference in Rust.
* **Actix-web:** 🕸️ A powerful, pragmatic, and extremely fast web framework for Rust.
* **Phi3-mini-4k-instruct:** 🧠 The language model serving as the core of this API.

## ⚙️ Getting Started

Follow these steps to set up the project locally:

1.  **Clone the Repository:** 
    ```bash
    git clone [https://github.com/d1pankarmedhi/Phi3-rust.git](https://github.com/d1pankarmedhi/Phi3-rust.git)
    cd Phi3-rust
    ```
2.  **Run the Application:** 
    ```bash
    cargo run --release
    ```

The Actix server will be available at `localhost:8080`.

## 📄 Usage

**Try out the endpoint:**

Make a `POST` request to `localhost:8080/query` with a **prompt**. 

**Request Body (JSON):**

```json
{
    "prompt": "QUESTION: What is 4 + 5?\nPlease answer this QUESTION under 50 words."
}
```
**Response (JSON):**

```json
{
    "text": "ANSWER: The sum of 4 and 5 is 9."
}
```
📝 Notes
This project is designed for high-performance Phi3 inference.
Performance may vary based on your system's hardware. 
Feel free to contribute to this project.
Make sure you have Rust and Cargo installed.








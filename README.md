# Phi3-rust
Serve Phi3 with Candle and Actix 🦀

Serving [Phi3 mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) with [Candle](https://github.com/huggingface/candle/tree/main) and [Actix-web](https://github.com/actix/actix-web) as RESTful API.

## Getting Started
Follow the steps to setup the project on local.

```bash
git clone https://github.com/d1pankarmedhi/Phi3-rust.git
cd Phi3-rust
cargo run --release
```

The Actix server can be found at `localhost:8080`. 

**Try out the endpoint**\
Make a `POST` request at `localhost:8080/query`  a **prompt**.
```JSON
// body
{
    "prompt": "QUESTION: What is 4 + 5?\nPlease answer this QUESTION under 50 words."
}
```
```JSON
// response
{
    "text": "ANSWER: The sum of 4 and 5 is 9."
}
```


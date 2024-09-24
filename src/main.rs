mod model;
use actix_web::{
    body::BoxBody, get, http::header::ContentType, post, web, App, Error, HttpRequest,
    HttpResponse, HttpServer, Responder,
};
use model::generation::Inference;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct UserQuery {
    prompt: String,
}

#[derive(Serialize)]
struct QueryResponse {
    text: String,
}
impl Responder for QueryResponse {
    type Body = BoxBody;
    fn respond_to(self, _req: &HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();

        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}
async fn query(query: web::Json<UserQuery>) -> impl Responder {
    let response =
        Inference::run(&query.prompt).unwrap_or("Error: Failed to run model".to_string());
    QueryResponse { text: response }
}

#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Server is UP!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(hello)
            .route("/query", web::post().to(query))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

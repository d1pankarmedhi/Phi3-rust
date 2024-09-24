use crate::model::utils::TokenOutputStream;
use crate::model::utils::{device, hub_load_safetensors};
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use tokenizers::Tokenizer;

pub struct TextGeneration {
    model: Phi3,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Phi3,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        let mut response = String::from("");

        println!("INFO: starting the inference loop");
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        print!("{prompt}");
        std::io::stdout().flush()?;
        let start_gen = std::time::Instant::now();
        let mut pos = 0;
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?.i((.., 0, ..))?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                if let Some(t) = self.tokenizer.decode_rest()? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                    response.push_str(t.as_str());
                }
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
                response.push_str(t.as_str());
            }
            pos += context_size;
        }
        let dt = start_gen.elapsed();
        println!(
            "\nINFO: {generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(response)
    }
}

struct Hyperparameters {
    cpu: bool,
    verbose_prompt: bool,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: u64,
    sample_len: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
}
impl Hyperparameters {
    fn default() -> Self {
        Self {
            cpu: true,
            verbose_prompt: true,
            temperature: Some(0.1),
            top_p: Some(0.8),
            seed: 666666,
            sample_len: 2000,
            repeat_penalty: 1.1,
            repeat_last_n: 32,
        }
    }
}

pub struct Inference {}

impl Inference {
    pub fn run(prompt: &str) -> Result<String> {
        let hyperparameters = Hyperparameters::default();
        println!(
            "temp: {}, repeat-penalty: {}, repeat-last-n: {}",
            hyperparameters.temperature.unwrap_or(0.0),
            hyperparameters.repeat_penalty,
            hyperparameters.repeat_last_n
        );

        // Load model
        let api = Api::new()?;
        let model_id = "microsoft/Phi-3-mini-4k-instruct".to_string();
        let revision = "main".to_string();
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let filenames = hub_load_safetensors(&repo, "model.safetensors.index.json")?;
        println!("INFO: Model retrieved!");

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let device = device(hyperparameters.cpu)?;
        let dtype = device.bf16_default_to_f32();

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let config_filename = repo.get("config.json")?;
        let config = std::fs::read_to_string(config_filename)?;
        let config: Phi3Config = serde_json::from_str(&config)?;
        let model = Phi3::new(&config, vb)?;

        println!("INFO: Model loaded!");

        // run model
        let mut pipeline = TextGeneration::new(
            model,
            tokenizer,
            hyperparameters.seed,
            hyperparameters.temperature,
            hyperparameters.top_p,
            hyperparameters.repeat_penalty,
            hyperparameters.repeat_last_n,
            hyperparameters.verbose_prompt,
            &device,
        );
        let response = pipeline.run(prompt, hyperparameters.sample_len)?;
        Ok(response)
    }
}

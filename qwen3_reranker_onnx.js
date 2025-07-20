import {
  AutoTokenizer,
  AutoModelForCausalLM,
  Tensor,
} from "@huggingface/transformers";

export class Qwen3RerankerONNX {
  constructor({
    modelPath = "zhiqing/Qwen3-Reranker-0.6B-ONNX",
    tokenizerDir = "zhiqing/Qwen3-Reranker-0.6B-ONNX",
    maxLength = 2048,
    dtype = "fp32",
    device = null,
  } = {}) {
    this.modelPath = modelPath;
    this.tokenizerDir = tokenizerDir;
    this.maxLength = maxLength;
    this.dtype = dtype;
    this.device = device;
    this.initialized = false;
  }

  async init() {
    this.tokenizer = await AutoTokenizer.from_pretrained(this.tokenizerDir, {
      padding_side: "left",
    });
    this.model = await AutoModelForCausalLM.from_pretrained(this.modelPath, {
      dtype: this.dtype,
      device: this.device,
    });
    this.prefix =
      "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
    this.suffix =
      "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const pre = await this.tokenizer(this.prefix, { add_special_tokens: false });
    const suf = await this.tokenizer(this.suffix, { add_special_tokens: false });
    this.prefixLen = pre.input_ids.dims[1];
    this.suffixLen = suf.input_ids.dims[1];
    this.defaultInstruction =
      "Given a web search query, retrieve relevant passages that answer the query";
    this.tokenFalseId = this.tokenizer.convert_tokens_to_ids("no");
    this.tokenTrueId = this.tokenizer.convert_tokens_to_ids("yes");
    this.initialized = true;
  }

  _formatInstruction(instruction, query, doc) {
    const inst = instruction ?? this.defaultInstruction;
    return `<Instruct>: ${inst}\n<Query>: ${query}\n<Document>: ${doc}`;
  }

  async _tokenize(pairs) {
    const texts = pairs.map((s) => this.prefix + s + this.suffix);
    const encoded = await this.tokenizer(texts, {
      padding: true,
      truncation: "longest_first",
      max_length: this.maxLength - this.prefixLen - this.suffixLen,
      add_special_tokens: false,
      return_tensor: true,
    });
    const seqLen = encoded.input_ids.dims[1];
    const batch = encoded.input_ids.dims[0];
    const singlePos = new BigInt64Array(seqLen);
    for (let i = 0n; i < BigInt(seqLen); ++i) {
      singlePos[Number(i)] = i;
    }
    const posData = new BigInt64Array(seqLen * batch);
    for (let b = 0; b < batch; ++b) {
      posData.set(singlePos, b * seqLen);
    }
    const position_ids = new Tensor("int64", posData, [batch, seqLen]);
    return {
      input_ids: encoded.input_ids,
      attention_mask: encoded.attention_mask,
      position_ids,
      batch,
    };
  }

  async infer(queries, documents, instruction = null) {
    if (!this.initialized) {
      await this.init();
    }
    if (queries.length === 1 && documents.length > 1) {
      queries = Array(documents.length).fill(queries[0]);
    } else if (queries.length !== documents.length) {
      throw new Error(
        "The number of queries must be 1 or equal to the number of documents."
      );
    }
    const pairs = queries.map((q, i) =>
      this._formatInstruction(instruction, q, documents[i])
    );
    const { input_ids, attention_mask, position_ids, batch } = await this._tokenize(pairs);
    const outputs = await this.model({
      input_ids,
      attention_mask,
      position_ids,
    });
    const seqLen = outputs.logits.dims[1];
    const last = outputs.logits.slice(null, seqLen - 1, null);
    const trueLogits = last.slice(null, null, [this.tokenTrueId, this.tokenTrueId + 1]).squeeze();
    const falseLogits = last.slice(null, null, [this.tokenFalseId, this.tokenFalseId + 1]).squeeze();
    const scoresYes = [];
    const scoresNo = [];
    for (let i = 0; i < batch; ++i) {
      const noLog = Number(falseLogits.data[i]);
      const yesLog = Number(trueLogits.data[i]);
      const maxLog = Math.max(noLog, yesLog);
      const expNo = Math.exp(noLog - maxLog);
      const expYes = Math.exp(yesLog - maxLog);
      const sum = expNo + expYes;
      scoresNo.push(expNo / sum);
      scoresYes.push(expYes / sum);
    }
    return { scores_yes: scoresYes, scores_no: scoresNo };
  }
}

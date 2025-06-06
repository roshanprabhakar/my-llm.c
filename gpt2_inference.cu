#define TESTING
#include "train_gpt2_fp32.cu"

#include <stdexcept>
#include <string.h>

// Let's write an inference server for gpt2_small using the kernels already included in this
// repository.

#define GPT2_EOT 50256

int main(int argc, char const **argv) {

	const char *system_prefix = 
		"I am a helpful chatbot tasked with answering user questions about the world. "
		"I am given the query '";
	const char *system_suffix = "', my response is the following: ";

	char user_prompt[50];
	user_prompt[sizeof(user_prompt)-1] = 0;

	int num_tokens = 50;
	for (int i = 1; i < argc; i += 2) {
		if (i + 1 >= argc) { throw std::runtime_error("Usage: gpt_inference [-n | --num-tokens INT]"); }
		if (argv[i][0] != '-') { throw std::runtime_error("Parameter must be passed via a flag."); }
		if (strlen(argv[i]) != 2) { throw std::runtime_error("Pass parameters via a dash '-' and a single character flag."); }
		if (argv[i][1] == 'n') { num_tokens = atoi(argv[i + 1]); }
		if (argv[i][1] == 'p') { 
			strncpy(user_prompt, argv[i + 1], sizeof(user_prompt)-1);
		}
	}

	char prompt[strlen(system_prefix) + strlen(user_prompt) + strlen(system_suffix)];
	strcpy(prompt, system_prefix);
	strcpy(prompt + strlen(system_prefix), user_prompt);
	strcpy(prompt + strlen(system_prefix) + strlen(user_prompt), system_suffix);

	printf("%s", prompt);

	return 0;


	// Set up the device.
	int deviceIdx = 0;
	cudaCheck(cudaSetDevice(deviceIdx));
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceIdx);
	printf("[System]\n");
	printf("Device %d: %s\n", deviceIdx, deviceProp.name);

	// Setup cuBLAS and cuBLASLt.
	cublasCheck(cublasCreate(&cublas_handle));
	// TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
	int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
	enable_tf32 = 0; // NOTE: disable TF32 for testing!!!
	printf("enable_tf32: %d\n", enable_tf32);
	cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
	cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
	cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

	// Build the GPT-2 model from a checkpoint.
	GPT2 model;
	gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

	// Build the tokenizer.
	Tokenizer tokenizer;
	tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

	// int C = model.config.channels;
	int B = 1;
	int T = 1024; // Maximum sequence length for gpt2_small.
	// int V = model.config.vocab_size;
	int Vp = model.config.padded_vocab_size;
	// int maxT = model.config.max_seq_len;
	// int L = model.config.num_layers;

	// Input tokens to feed the model.
	int *gen_tokens = reinterpret_cast<int *>(mallocCheck(B * T * sizeof(int)));

	// Host memory to hold logits.
	float *cpu_logits = reinterpret_cast<float *>(mallocCheck(B * T * Vp * sizeof(float)));

	for (int i = 0; i < B * T; ++i) { gen_tokens[i] = GPT2_EOT; }
	for (int t = 1; t < num_tokens; ++t) {
		gpt2_forward(&model, gen_tokens, NULL, B, T, 0);
		float *device_logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
		cudaCheck(cudaMemcpy(cpu_logits, device_logits, model.config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

		// Get logit corresponding to maximum activation.
		int max_logit_i = 0;
		for (int i = 1; i < model.config.vocab_size; ++i) {
			if (cpu_logits[i] > cpu_logits[max_logit_i]) max_logit_i = i;
		}

		// Add the generated token tot he end of the input sequence.
		gen_tokens[t] = max_logit_i;
	}	

	// Print the generated sequence.
	int eot_count = 0;
	for (int i = 0; i < B && eot_count < 2; ++i) {
		for (int t = 0; t < T && eot_count < 2; ++t) {
			if (gen_tokens[T * i + t] == GPT2_EOT) { eot_count++; }
			if (tokenizer.init_ok) {
				printf("%s", tokenizer_decode(&tokenizer, gen_tokens[T * i + t]));
			} else {
				printf("%d ", gen_tokens[T * i + t]);
			}
		}
		printf("<EOS>\n");
	}

	return 0;
}


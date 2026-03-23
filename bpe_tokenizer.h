#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <stdio.h>

#define BPE_MAX_VOCAB_SIZE 8192
#define BPE_MAX_TOKEN_LEN 64
#define BPE_MAX_MERGES 8000

typedef struct {
  int first;
  int second;
  int result;
} BPEMerge;

typedef struct {

  char** vocab;
  int vocab_size;

  BPEMerge* merges;
  int num_merges;

  int pad_id;
  int unk_id;
  int bos_id;
  int eos_id;

  int char_to_id[256];
} BPETokenizer;

BPETokenizer* bpe_tokenizer_create(void);

void bpe_tokenizer_free(BPETokenizer* tok);

int bpe_train(BPETokenizer* tok, const char* text, int num_merges);

int bpe_train_from_file(BPETokenizer* tok, const char* filepath, int num_merges);

int bpe_save_vocab(BPETokenizer* tok, const char* filepath);

int bpe_load_vocab(BPETokenizer* tok, const char* filepath);

int* bpe_encode(BPETokenizer* tok, const char* text, int* out_len,
        int add_bos, int add_eos);

char* bpe_decode(BPETokenizer* tok, int* ids, int len);

const char* bpe_decode_token(BPETokenizer* tok, int id);

int bpe_vocab_size(BPETokenizer* tok);

void bpe_print_stats(BPETokenizer* tok);

void bpe_print_tokens(BPETokenizer* tok, int* ids, int len);

#endif

CC = gcc
LDFLAGS = -I. -ldl -lpthread -lm
ifeq ($(build),release)
	CFLAGS = -O3
	LDFLAGS += -DNDEBUG=1
else
	CFLAGS = -Og -g
endif
CFLAGS += -std=gnu99 -Wall -Wextra -Werror -pedantic
RM = rm -rf

OBJECTS = \
	attention.o bpe_tokenizer.o embedding.o ffn.o kv_cache.o \
	layernorm.o math_ops.o model.o tensor.o transformer.o

OBJECTS := $(addprefix objects/,$(OBJECTS))

TRAIN_OBJECTS = loss.o backward.o optimizer.o
TRAIN_OBJECTS := $(addprefix objects/,$(TRAIN_OBJECTS))

all: objects $(OBJECTS) $(TRAIN_OBJECTS)

chat: objects/chat.o $(OBJECTS) $(TRAIN_OBJECTS)
ifeq ($(build),release)
	@echo "Build release '$@' executable ..."
else
	@echo "Build '$@' executable ..."
endif
	@$(CC) objects/chat.o $(OBJECTS) $(TRAIN_OBJECTS) -o $@ $(LDFLAGS) -DLLM_CHAT=1
	@$(RM) objects/chat.o

llm: objects/llm.o $(OBJECTS) $(TRAIN_OBJECTS)
ifeq ($(build),release)
	@echo "Build release '$@' executable ..."
else
	@echo "Build '$@' executable ..."
endif
	@$(CC) objects/llm.o $(OBJECTS) $(TRAIN_OBJECTS) -o $@ $(LDFLAGS)
	@$(RM) objects/llm.o

objects:
	@echo "Create 'objects' folder ..."
	@mkdir -p objects

objects/chat.o: llm.c
ifeq ($(build),release)
	@echo "Build release '$@' object ..."
else
	@echo "Build '$@' object ..."
endif
	@$(CC) -c $(CFLAGS) $< -o $@ $(LDFLAGS) -DLLM_CHAT=1

objects/%.o: %.c
ifeq ($(build),release)
	@echo "Build release '$@' object ..."
else
	@echo "Build '$@' object ..."
endif
	@$(CC) -c $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	@echo "Cleanup ..."
	@$(RM) $(OBJECTS) $(TRAIN_OBJECTS) chat llm

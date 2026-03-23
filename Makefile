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

objects:
	@echo "Create 'objects' folder ..."
	@mkdir -p objects

objects/%.o: %.c
ifeq ($(build),release)
	@echo "Build release '$@' object ..."
else
	@echo "Build '$@' object ..."
endif
	@$(CC) -c $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	@echo "Cleanup ..."
	@$(RM) $(OBJECTS)

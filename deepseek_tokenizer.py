import transformers

# 加载tokenizer
chat_tokenizer_dir = "./"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    chat_tokenizer_dir, trust_remote_code=True
)

# 增加一些额外的方法，确保与我们的网站兼容

# 如果原始tokenizer没有convert_ids_to_tokens方法，添加这个方法
if not hasattr(tokenizer, 'convert_ids_to_tokens'):
    def convert_ids_to_tokens(self, ids):
        """将token ID转换为文本表示"""
        result = []
        for token_id in ids:
            # 尝试通过逆向查找词汇表来找到token文本
            for token, tid in self.get_vocab().items():
                if tid == token_id:
                    result.append(token)
                    break
            else:
                # 如果找不到对应的token文本，使用ID作为占位符
                result.append(f"<{token_id}>")
        return result
    
    # 将方法添加到tokenizer
    tokenizer.convert_ids_to_tokens = convert_ids_to_tokens.__get__(tokenizer)

# 如果原始tokenizer没有vocab_size属性，添加此属性
if not hasattr(tokenizer, 'vocab_size'):
    try:
        tokenizer.vocab_size = len(tokenizer.get_vocab())
    except:
        tokenizer.vocab_size = 0  # 默认值

# 如果原始tokenizer没有model_max_length属性，添加此属性
if not hasattr(tokenizer, 'model_max_length'):
    tokenizer.model_max_length = 4096  # 默认值

# 如果原始tokenizer没有name_or_path属性，添加此属性
if not hasattr(tokenizer, 'name_or_path'):
    tokenizer.name_or_path = "./自定义DeepSeek分词器"

# 如果原始tokenizer没有all_special_tokens属性，添加此属性
if not hasattr(tokenizer, 'all_special_tokens'):
    tokenizer.all_special_tokens = []  # 默认值

# 如果原始tokenizer没有special_tokens_map属性，添加此属性
if not hasattr(tokenizer, 'special_tokens_map'):
    tokenizer.special_tokens_map = {}  # 默认值

# 如果需要测试分词器功能，可以取消下面注释
if __name__ == "__main__":
    result = tokenizer.encode("Hello!")
    print(f"Tokens: {result}")
    
    try:
        token_texts = tokenizer.convert_ids_to_tokens(result)
        print(f"Token texts: {token_texts}")
    except Exception as e:
        print(f"无法获取token文本: {e}")
    
    try:
        print(f"Vocabulary size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"无法获取词汇量: {e}")
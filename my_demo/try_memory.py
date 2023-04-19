from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from config import OPENAI_API_KEY

llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# llm = OpenAI(temperature=0) # Can be any valid LLM
_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)

import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS


embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY).embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)
memory.save_context({"input": "My favorite food is pizza"}, {"output": "thats good to know"})
memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"})


conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    # We set a very low max_token_limit for the purposes of testing.
    memory=memory,
    verbose=True
)
ai_rsp = conversation_with_summary.predict(input="Hi, my name is Perry, what's up?")
ai_rsp = conversation_with_summary.predict(input="what's my favorite sport?")
print(f"AI回复: {ai_rsp}")
# print(memory.load_memory_variables({}))
# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory=ConversationSummaryBufferMemory(llm=OpenAI(openai_api_key=OPENAI_API_KEY), max_token_limit=10)
# )
# print(conversation.prompt.template)
# user_inputs = ["hi there!",
#                "can you tell me what you can do",
#                "give me some math problems"]
# for user_input in user_inputs:
#     ai_rsp = conversation.predict(input=user_input)
#     print(f"AI回复:", ai_rsp)

# memory = ConversationBufferMemory(return_messages=False)
# memory.save_context({"input": "你好"}, {"ouput": "你好，怎么了，我是AI金融智能问答机器人，有什么可以帮您"})
# memory.save_context({"input": "你能做什么事"}, {"ouput": "我可以解答个股相关问题，包括但不限于资讯、行情、财务类问题"})
# print(memory.load_memory_variables({}))
# '''
# ['Human: 你好',
#  'AI: 你好，怎么了，我是AI金融智能问答机器人，有什么可以帮您',
#  'Human: 你能做什么事',
#  'AI: 我可以解答个股相关问题，包括但不限于资讯、行情、财务类问题']
# '''
# print(memory.memory_variables)
# print()
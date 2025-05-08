from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

plantilla = """¿Cuál es la ciudad más popular en {country} para los turistas?
Solo devuelve el nombre de la ciudad."""

primer_prompt = PromptTemplate(
    input_variables=["country"],
    template=plantilla
)

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o"
)
llm2 = ChatOpenAI(
    temperature=0,
    model="gpt-4"
)

cadena_uno = LLMChain(llm=llm, prompt=primer_prompt, verbose=True)

segundo_prompt = PromptTemplate(
    input_variables=["city"],
    template=("¿Cuáles son las tres principales cosas que hacer en esta ciudad: {city} para los turistas? \
Solo devuelve la respuesta como tres puntos")
)
cadena_dos = LLMChain(llm=llm2, prompt=segundo_prompt, verbose=True)

cadena_general = SimpleSequentialChain(chains=[cadena_uno, cadena_dos], verbose=True)
repuesta_final = cadena_general.invoke("Canadá")

print(repuesta_final)


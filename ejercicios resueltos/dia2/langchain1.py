from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

load_dotenv()

chat = ChatOpenAI(
    temperature=1,
    model="gpt-4o"
)
# response = chat.invoke("Como de guapos son los alumnos de keepcoding?")
# print(response.content)
customer_email = """
Arrr, estoy furioso porque la tapa de mi licuadora salió volando y salpicó las paredes de mi cocina con batido. Y para empeorar las cosas, la garantía no cubre el costo de limpiar mi cocina. ¡Necesito tu ayuda ahora mismo, amigo!
"""
style = " Español de España tranquilo y relajado"
prompt = f"""Traduce el texto
delimitado por triples comillas invertidas
a un estilo que sea {style}.
texto: {customer_email}
"""
# response = chat.invoke(prompt)
# print(response.content)

template_string = """Traduce el texto
delimitado por triples comillas invertidas
a un estilo que sea {style}.
texto: ```{text}```
"""

prompt = ChatPromptTemplate.from_template(template_string)
customer_message = prompt.format_messages(
    text=customer_email,
    style=style
)

service_reply = """Hola, cliente,
la garantía no cubre
los gastos de limpieza de su cocina
porque es su culpa que
haya usado mal su licuadora
al olvidar poner la tapa antes de
encender la licuadora.
¡Qué mala suerte! ¡Adiós!
"""

service_style_pirate = """\
un tono educado \
que habla en español pirata\
"""
service_messages = prompt.format_messages(
    style=service_style_pirate,
    text=service_reply)

#service_response = chat(service_messages)
#print(service_response.content)

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(
    name="gift",
    description=("""¿Fue el artículo comprado
                             como un regalo para otra persona?
                             Responde True si es así,
                             False si no o si no se sabe.""")
)
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="""¿Cuántos días
                                      tardó en llegar el producto?
                                      Si esta información no se encuentra,
                                      escribe -1.""")
price_value_schema = ResponseSchema(name="price_value",
                                    description="""Extrae cualquier
                                    oración sobre el valor o
                                    precio, y escríbelas en una
                                    lista separada por comas en Python.""")
response_schemas = [gift_schema,
                    delivery_days_schema,
                    price_value_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas )
format_instructions = output_parser.get_format_instructions()

customer_review = """
Este soplador de hojas es bastante increíble. Tiene cuatro configuraciones: soplador de velas,
brisa suave, ciudad ventosa y tornado. Llegó en dos días, justo a tiempo para el regalo de aniversario de mi esposa.
Creo que a mi esposa le gustó tanto que se quedó sin palabras. Hasta ahora, soy el único que lo ha usado,
y lo he estado usando cada dos mañanas para limpiar las hojas de nuestro césped. Es un poco más caro que otros sopladores
de hojas en el mercado, pero creo que vale la pena por las características adicionales.
"""

review_template_2 = """\
Para el siguiente texto, extrae la siguiente información:

gift: ¿Fue el artículo comprado como un regalo para otra persona? \
Responde True si es así, False si no o si no se sabe.

delivery_days: ¿Cuántos días tardó en llegar el producto? \
Si esta información no se encuentra, escribe -1.

price_value: Extrae cualquier oración sobre el valor o precio, \
y escríbelas en una lista separada por comas en Python.

texto: {text}

{format_instructions}"""

prompt_2 = ChatPromptTemplate.from_template(review_template_2)
messages = prompt_2.format_messages(text=customer_review, format_instructions=format_instructions)
response = chat.invoke(messages)
#print(response.content)
out_dir = output_parser.parse(response.content)
#print(out_dir)
print(f'"gift":{out_dir["gift"]}')
print("delivery_days:", out_dir.get('delivery_days'))
print("price_value:", out_dir.get('price_value'))


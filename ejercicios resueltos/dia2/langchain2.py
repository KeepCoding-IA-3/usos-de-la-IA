import json

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()


# crear una tool
@tool
def calculador_precio_final(precio_base: Annotated[float, "Precio sin IVA en euros"],
                            porcentaje_de_iva: Annotated[float, "Porcentaje de IVA, como 21 para 21%"]) -> float:
    """Calcula el precio final con IVA incluido."""
    return round(precio_base * (1 + porcentaje_de_iva / 100), 2)


llm = ChatOpenAI(
    temperature=1,
    model="gpt-4"
)
message = HumanMessage(content="Calcula el precio final si el precio base es 175 y el IVA es 21%")
# unimos la tool con el modelo
llm_with_tools = llm.bind_tools([calculador_precio_final])

#probarlo
respuesta = llm_with_tools.invoke([message])

if respuesta.tool_calls:
    args = respuesta.tool_calls[0]["args"]
    nombre_funcion = respuesta.tool_calls[0]["name"]

    print(f"precio_base: {args['precio_base']}, porcentaje_iva: {args['porcentaje_de_iva']}")

    # Ejecutar la tool
    resultado = calculador_precio_final.invoke(args)
    print("Resultado tool:", resultado)

    # Mensaje con la respuesta de la tool
    tool_response = ToolMessage(tool_call_id=respuesta.tool_calls[0]["id"], content=str(resultado))

    # Segunda llamada, pero usando la respuesta anterior y el resultado de la tool
    respuesta_final = llm_with_tools.invoke([respuesta, tool_response])
    print("Respuesta final del modelo:", respuesta_final.content)


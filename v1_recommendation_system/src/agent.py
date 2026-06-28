# The agentic loop: connects the LLM (OpenAI) to the tools defined in tools.py

import json
import os

from openai import OpenAI

from tools import TOOL_SCHEMAS, AVAILABLE_FUNCTIONS

#from dotenv import load_dotenv
#load_dotenv()

MODEL = "gpt-4o-mini"
MAX_TURNS = 8  # stop infinite agent loops

SYSTEM_PROMPT = """\
És um assistente de recomendação de produtos Nintendo Switch (consolas, jogos e acessórios).

Tens acesso a ferramentas para consultar o catálogo de produtos e os padrões de compra (que produtos são frequentemente comprados juntos).

Regras importantes:
- Usa SEMPRE as ferramentas para obter informação factual sobre produtos. Nunca inventes nomes de produtos, preços ou disponibilidade.
- Tens três formas de recomendar:
  * get_recommendations_npmi -> MOTOR PRINCIPAL. Recomenda por força de associação (NPMI, Market Basket Analysis), corrigindo o viés de popularidade \
e inclui o cosseno como validação cruzada. Usa esta por defeito para recomendar produtos relacionados/complementares com ligação genuína.
  * get_cooccurring_products -> produtos COMPRADOS JUNTOS por contagem bruta. Usa só se quiseres a frequência absoluta de co-compra (não corrigida por popularidade).
  * get_similar_products -> produtos PARECIDOS (perfil de cliente semelhante, via cosseno). Usa quando o utilizador gostou de algo e quer uma alternativa do mesmo estilo.
- Se o pedido do utilizador não tiver nada a ver com produtos Nintendo Switch (ex: comida, outros assuntos), explica educadamente que só podes ajudar com \
recomendações de produtos Nintendo Switch.
- Quando o utilizador pedir para excluir uma "família" ou franchise de produtos, usa a ferramenta exclude_franchise.
- Quando o utilizador mencionar uma loja específica, usa filter_by_store para confirmar disponibilidade nessa loja antes de recomendar.
- Quando o utilizador mencionar a idade de alguém, considera o campo min_age dos produtos para garantir adequação etária.
- Sê transparente: se não tiveres informação suficiente para responder com confiança, diz isso ao utilizador em vez de adivinhar.

FORMATO DAS RESPOSTAS (muito importante):
- Por defeito, ao recomendar produtos mostra APENAS o nome do produto e o seu link (campo 'url'). NÃO incluas o tipo, a descrição, a categoria, a idade mínima, \
as vendas nem os scores (npmi/cosseno) na resposta ao utilizador.
- Apresenta cada recomendação de forma limpa, ex: "Nome do Produto" com o link.
- Usa a informação extra (idade mínima, loja, franchise) para DECIDIR o que recomendar (filtrar, excluir, adequar à idade), mas NÃO a mostres na resposta.
- SÓ quando o utilizador pedir explicitamente mais informação ou detalhes sobre um produto específico é que deves então dar o tipo, descrição, idade mínima, etc. \
(usando get_product_info para esse produto).
"""


def run_agent(user_query: str, history: list | None = None, verbose: bool = True) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"  [tool call] {function_name}({function_args})")

            function = AVAILABLE_FUNCTIONS.get(function_name)
            if function is None:
                result = {"error": f"Função desconhecida: {function_name}"}
            else:
                result = function(**function_args)

            if verbose:
                print(f"  [tool result] {result}\n")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    return (
        "Não consegui chegar a uma resposta definitiva dentro do limite de passos permitido. Tenta reformular a pergunta de forma mais específica."
    )


if __name__ == "__main__":
    test_queries = [
        "I want a pepperoni pizza with extra cheese please.",
        (
            "I want to buy a game for my nephew, at Store A, who is 5 years old. "
            "We loved Super Mario Odyssey, but I cannot buy a game from this "
            "family as he already has all Super Mario games."
        ),
    ]

    for q in test_queries:
        print(f"\n Query: {q} ")
        answer = run_agent(q)
        print(f"\nResposta final: {answer}\n{'-' * 60}")

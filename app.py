"""
Simulador de Crédito para a Primeira Moto
=========================================
Stack: Python + Streamlit (interface) + TensorFlow (indicador estimativo opcional).

Arquitetura (separação de responsabilidades, no espírito do MVC):
  - Modelo      -> dataclasses SolicitacaoCredito e ResultadoAnalise
  - Regras      -> classe AnalisadorCredito (100% determinística, sem IA)
  - Estimativa  -> função com TensorFlow (apenas informativa, NÃO decide o crédito)
  - Visão       -> funções render_* do Streamlit

Para executar:
    pip install streamlit tensorflow numpy
    streamlit run app.py
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Optional

import streamlit as st

# ----------------------------------------------------------------------
# 1) CONSTANTES DE NEGÓCIO
# Ficam no topo para que uma mudança de política do banco seja feita
# em um único lugar.
# ----------------------------------------------------------------------
IDADE_MINIMA = 18
IDADE_MAXIMA = 70
LIMITE_COMPROMETIMENTO = Decimal("0.30")  # parcela <= 30% da renda líquida


# ----------------------------------------------------------------------
# 2) MODELO DE DADOS
# ----------------------------------------------------------------------
class Status(Enum):
    """Possíveis desfechos da análise."""
    APROVADO = "aprovado"
    NEGADO_IDADE = "negado_idade"
    NEGADO_RENDA = "negado_renda"


@dataclass(frozen=True)
class SolicitacaoCredito:
    """Dados informados pelo cliente. Decimal evita erros de arredondamento
    com dinheiro (ex.: 0.1 + 0.2 != 0.3 em float)."""
    idade: int
    renda_liquida: Decimal
    parcela_desejada: Decimal


@dataclass(frozen=True)
class ResultadoAnalise:
    """Resultado da análise, pronto para a interface exibir."""
    status: Status
    parcela_maxima: Decimal                 # sempre 30% da renda
    comprometimento: Decimal                # parcela / renda (em %)
    parcela_sugerida: Optional[Decimal] = None  # só preenchida se negado por renda


# ----------------------------------------------------------------------
# 3) REGRAS DE NEGÓCIO (determinísticas)
# ----------------------------------------------------------------------
class AnalisadorCredito:
    """Aplica, nesta ordem, as regras do banco."""

    @staticmethod
    def _truncar_centavos(valor: Decimal) -> Decimal:
        # Truncamos (não arredondamos para cima) para nunca sugerir
        # uma parcela maior que o limite de 30%.
        return valor.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def analisar(self, s: SolicitacaoCredito) -> ResultadoAnalise:
        parcela_maxima = self._truncar_centavos(s.renda_liquida * LIMITE_COMPROMETIMENTO)
        comprometimento = (s.parcela_desejada / s.renda_liquida) * 100

        # Regra 1 - Idade: tem prioridade. Se falhar aqui, a alternativa de
        # parcela NÃO é oferecida (a negativa não é "exclusivamente" por renda).
        if s.idade < IDADE_MINIMA or s.idade > IDADE_MAXIMA:
            return ResultadoAnalise(Status.NEGADO_IDADE, parcela_maxima, comprometimento)

        # Regra 2 - Compromisso de renda: parcela <= 30% da renda.
        if s.parcela_desejada > s.renda_liquida * LIMITE_COMPROMETIMENTO:
            return ResultadoAnalise(
                Status.NEGADO_RENDA, parcela_maxima, comprometimento,
                parcela_sugerida=parcela_maxima,
            )

        return ResultadoAnalise(Status.APROVADO, parcela_maxima, comprometimento)


# ----------------------------------------------------------------------
# 4) INDICADOR ESTIMATIVO COM TENSORFLOW (opcional e informativo)
# Uma rede neural minúscula, treinada com dados sintéticos, que estima
# a "folga" do orçamento. A DECISÃO continua sendo das regras acima.
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def carregar_modelo_folga():
    """Treina (uma única vez, graças ao cache) um modelo que aprende a
    relação entre comprometimento de renda e chance de aprovação."""
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        return None  # app funciona normalmente sem TensorFlow

    rng = np.random.default_rng(42)
    comprometimento = rng.uniform(0.0, 0.8, size=(4000, 1)).astype("float32")
    rotulo = (comprometimento <= 0.30).astype("float32")  # mesma regra do banco

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    modelo.compile(optimizer="adam", loss="binary_crossentropy")
    modelo.fit(comprometimento, rotulo, epochs=40, batch_size=64, verbose=0)
    return modelo


def estimar_chance(comprometimento_pct: Decimal) -> Optional[float]:
    """Retorna a probabilidade (0-1) estimada pelo modelo, ou None."""
    modelo = carregar_modelo_folga()
    if modelo is None:
        return None
    import numpy as np
    x = np.array([[min(float(comprometimento_pct) / 100, 0.8)]], dtype="float32")
    return float(modelo.predict(x, verbose=0)[0][0])


# ----------------------------------------------------------------------
# 5) FORMATAÇÃO
# ----------------------------------------------------------------------
def brl(valor: Decimal) -> str:
    """Formata como moeda brasileira: 1234.5 -> 'R$ 1.234,50'."""
    texto = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {texto}"


# ----------------------------------------------------------------------
# 6) INTERFACE (VISÃO)
# ----------------------------------------------------------------------
CSS = """
<style>
.block-container {max-width: 760px; padding-top: 2rem;}
.hero {background: linear-gradient(135deg,#eef4ff,#f7f9fc); border-radius: 18px;
       padding: 1.4rem 1.6rem; margin-bottom: 1.2rem; border: 1px solid #e3eaf6;}
.hero h1 {margin: 0 0 .3rem 0; font-size: 1.8rem; color:#1f3a5f;}
.hero p {margin: 0; color:#4a5b73;}
.card {border-radius: 16px; padding: 1.3rem 1.5rem; margin-top: 1rem; line-height: 1.55;}
.card h3 {margin-top: 0;}
.ok   {background:#e8f8ee; border:1px solid #8fd6a7; color:#14532d;}
.warn {background:#fff1e6; border:1px solid #f4b183; color:#7c2d12;}
.bad  {background:#fdeaea; border:1px solid #ef9a9a; color:#7f1d1d;}
.sug  {margin-top:.8rem; padding:.8rem 1rem; background:#ffffffb3;
       border-radius:12px; font-weight:600;}
</style>
"""


def render_resultado(r: ResultadoAnalise, s: SolicitacaoCredito) -> None:
    """Mostra o card de resultado conforme o status."""
    if r.status is Status.APROVADO:
        st.balloons()
        st.markdown(
            f"""<div class="card ok"><h3>🎉 Crédito pré-aprovado! Sua primeira moto está logo ali!</h3>
            Parabéns! Uma parcela de <b>{brl(s.parcela_desejada)}</b> usa
            <b>{r.comprometimento:.1f}%</b> da sua renda, dentro do limite saudável de 30%.
            Isso mostra organização financeira. Este resultado é uma simulação; a
            aprovação final depende da análise completa do banco.</div>""",
            unsafe_allow_html=True,
        )

    elif r.status is Status.NEGADO_IDADE:
        st.markdown(
            f"""<div class="card bad"><h3>Não foi possível aprovar desta vez</h3>
            Pela política do banco, o financiamento é concedido para pessoas entre
            <b>{IDADE_MINIMA} e {IDADE_MAXIMA} anos</b>. Você informou <b>{s.idade} anos</b>,
            então a simulação não pode seguir. Se for o caso, um responsável
            maior de idade pode ser a melhor alternativa. Consulte uma agência
            para conhecer outras opções.</div>""",
            unsafe_allow_html=True,
        )

    else:  # NEGADO_RENDA
        st.markdown(
            f"""<div class="card warn"><h3>Quase lá! A parcela ficou acima do ideal</h3>
            A parcela de <b>{brl(s.parcela_desejada)}</b> compromete
            <b>{r.comprometimento:.1f}%</b> da sua renda, e o limite do banco é 30%.
            Manter as parcelas abaixo desse teto protege o seu orçamento para
            moradia, alimentação e imprevistos.
            <div class="sug">Sugerimos uma parcela de no máximo
            {brl(r.parcela_sugerida)} ou aumentar o prazo do financiamento.</div></div>""",
            unsafe_allow_html=True,
        )
        st.caption("💡 Dica: prazos maiores reduzem a parcela, mas aumentam o total de juros pagos. "
                   "Uma entrada maior também ajuda a baixar o valor financiado.")

    # Indicador estimativo (TensorFlow) - só informativo
    chance = estimar_chance(r.comprometimento)
    if chance is not None and r.status is not Status.NEGADO_IDADE:
        st.progress(min(max(chance, 0.0), 1.0),
                    text=f"Indicador estimado de folga no orçamento: {chance * 100:.0f}% "
                         "(apenas ilustrativo, não altera o resultado)")


def main() -> None:
    st.set_page_config(page_title="Simulador - Primeira Moto", page_icon="🏍️", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """<div class="hero"><h1>🏍️ Simulador de Crédito - Primeira Moto</h1>
        <p>Descubra em segundos se a parcela que você imaginou cabe no seu bolso.</p></div>""",
        unsafe_allow_html=True,
    )

    # st.form agrupa os campos: a análise só roda ao clicar no botão,
    # sem recarregar a página inteira manualmente.
    with st.form("form_credito"):
        idade = st.number_input("Idade (anos)", min_value=0, max_value=120, value=25, step=1)
        renda = st.number_input("Renda mensal líquida (R$)", min_value=0.0,
                                value=3000.0, step=100.0, format="%.2f")
        parcela = st.number_input("Valor da parcela desejada (R$)", min_value=0.0,
                                  value=800.0, step=50.0, format="%.2f")
        enviado = st.form_submit_button("Simular crédito", use_container_width=True, type="primary")

    if not enviado:
        st.info("Preencha os dados acima e clique em **Simular crédito**.")
        return

    # Validação de entrada: renda zero causaria divisão por zero.
    if renda <= 0 or parcela <= 0:
        st.warning("Informe uma renda e uma parcela maiores que zero para simular.")
        return

    solicitacao = SolicitacaoCredito(
        idade=int(idade),
        renda_liquida=Decimal(str(renda)),
        parcela_desejada=Decimal(str(parcela)),
    )
    render_resultado(AnalisadorCredito().analisar(solicitacao), solicitacao)


if __name__ == "__main__":
    main()

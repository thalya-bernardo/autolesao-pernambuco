# ============================================================
# CENÁRIO EPIDEMIOLÓGICO DE AUTOLESÕES EM ADOLESCENTES
# PERNAMBUCO | SINAN | 10–19 ANOS | 2014–2024
# ============================================================

import os
import re
import base64
import zipfile
import tempfile
import warnings
from math import sqrt

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Cenário Epidemiológico de Autolesões em Adolescentes em Pernambuco",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

AZUL = "#173B6C"
AZUL2 = "#3E6FA8"
VERMELHO = "#E52B35"

FEMININO = "#C44E8B"
MASCULINO = "#2878B5"

IDADE_10_14 = "#2A9D8F"
IDADE_15_19 = "#E9A23B"

REC_SIM = "#D95F02"
REC_NAO = "#4C78A8"

ESCALA_MAGNITUDE = [
    [0.00, "#FFF4D6"],
    [0.25, "#FDCB6E"],
    [0.50, "#F28E2B"],
    [0.75, "#E15759"],
    [1.00, "#A61B29"],
]

CORES_RACA = {
    "Parda": "#9C755F",
    "Branca": "#59A14F",
    "Preta": "#4E4E4E",
    "Amarela": "#EDC948",
    "Indígena": "#B07AA1",
}

SEXO_VALIDO = [
    "Feminino",
    "Masculino",
]

RACA_VALIDA = [
    "Parda",
    "Branca",
    "Preta",
    "Amarela",
    "Indígena",
]

RECORRENCIA_VALIDA = [
    "Sim",
    "Não",
]

ESCOLARIDADE_VALIDA = [
    "Analfabeto a 4ª série incompleta do EF",
    "4ª série completa do EF",
    "5ª a 8ª série incompleta do EF",
    "Ensino fundamental completo",
    "Ensino médio incompleto",
    "Ensino médio completo",
    "Educação superior incompleta",
    "Educação superior completa",
]

METODOS = {
    "AG_FORCA": "Força corporal/espancamento",
    "AG_ENFOR": "Enforcamento",
    "AG_OBJETO": "Objeto contundente",
    "AG_CORTE": "Objeto perfurocortante",
    "AG_QUENTE": "Substância/objeto quente",
    "AG_ENVEN": "Envenenamento/intoxicação",
    "AG_FOGO": "Arma de fogo",
    "AG_AMEACA": "Ameaça",
    "AG_OUTROS": "Outros",
}

ORDEM_N_METODOS = [
    "1 método",
    "2 métodos",
    "3 ou mais métodos",
]


# ============================================================
# 2. CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1900px;
    padding-top: 1.1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 4rem;
}

/* ==========================================================
   CABEÇALHO
   ========================================================== */

.header-logo-box {
    width: 100%;
    height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: visible;
}

.header-logo-box img {
    width: 105px !important;
    height: 105px !important;
    object-fit: contain !important;
    display: block;
    margin: 0 auto;
}

.header-central {
    width: 100%;
    text-align: center;
    padding: 2px 0 8px 0;
    margin: 0;
    overflow: visible;
}

.instituicao {
    width: 100%;
    text-align: center;
    color: #173B6C;
    font-family: Arial, sans-serif;
    font-size: 12px;
    font-weight: 800;
    line-height: 1.25;
    margin: 0 0 4px 0;
    white-space: nowrap;
}

.curso {
    width: 100%;
    text-align: center;
    color: #5D6875;
    font-family: Arial, sans-serif;
    font-size: 10.5px;
    font-weight: 600;
    line-height: 1.25;
    margin: 0 0 12px 0;
    white-space: nowrap;
}

.titulo-principal {
    width: 100%;
    text-align: center;
    color: #173B6C;
    font-family: Arial, sans-serif;
    font-size: clamp(15px, 1.45vw, 25px);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.15px;
    margin: 0 auto 10px auto;
    white-space: nowrap;
    overflow: visible;
}

.subtitulo {
    width: 100%;
    max-width: 1350px;
    text-align: center;
    color: #5C6673;
    font-family: Arial, sans-serif;
    font-size: 11.5px;
    font-weight: 400;
    line-height: 1.35;
    margin: 0 auto;
}

.header-bottom-space {
    height: 18px;
}

/* ==========================================================
   SEÇÕES
   ========================================================== */

.titulo-secao {
    font-size: 23px;
    font-weight: 800;
    color: #173B6C;
    margin-top: 34px;
    margin-bottom: 5px;
    border-bottom: 2px solid #E6E9ED;
    padding-bottom: 7px;
}

.descricao-secao {
    color: #68727D;
    font-size: 13px;
    margin-bottom: 16px;
}

/* ==========================================================
   MÉTRICAS
   ========================================================== */

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E6EB;
    border-radius: 12px;
    padding: 12px 14px;
    min-height: 105px;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
}

div[data-testid="stMetricLabel"] {
    font-size: 13px;
}

div[data-testid="stMetricValue"] {
    font-size: 27px;
    font-weight: 750;
    color: #173B6C;
}

/* ==========================================================
   SIDEBAR E COLUNAS
   ========================================================== */

[data-testid="stSidebar"] {
    border-right: 1px solid #E4E7EB;
}

[data-testid="column"] {
    padding-left: 0.30rem;
    padding-right: 0.30rem;
}

/* ==========================================================
   RESPONSIVIDADE
   ========================================================== */

@media (max-width: 1500px) {

    .titulo-principal {
        font-size: 17px;
    }

    .instituicao {
        font-size: 10.5px;
    }

    .curso {
        font-size: 9px;
    }

    .subtitulo {
        font-size: 10.5px;
    }

    .header-logo-box {
        height: 105px;
    }

    .header-logo-box img {
        width: 90px !important;
        height: 90px !important;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. FUNÇÕES AUXILIARES
# ============================================================

def localizar_arquivo(opcoes):
    for nome in opcoes:
        if os.path.exists(nome):
            return nome

    raise FileNotFoundError(
        "Nenhum destes arquivos foi encontrado: "
        + ", ".join(opcoes)
    )


def idade_em_anos(valor):
    if pd.isna(valor):
        return np.nan

    try:
        valor = int(float(valor))
    except (ValueError, TypeError):
        return np.nan

    unidade = valor // 1000
    numero = valor % 1000

    if unidade == 4:
        return numero

    return np.nan


def codigo_municipio_6d(valor):
    if pd.isna(valor):
        return pd.NA

    try:
        return str(int(float(valor))).zfill(6)[:6]

    except (ValueError, TypeError):
        texto = re.sub(r"\D", "", str(valor))

        if len(texto) >= 6:
            return texto[:6]

        return pd.NA


def numero_br(valor, casas=0):
    if pd.isna(valor):
        return "—"

    texto = f"{valor:,.{casas}f}"

    return (
        texto
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def percentual_br(valor, casas=1):
    if pd.isna(valor):
        return "—"

    return f"{numero_br(valor, casas)}%"


def decodificar_sexo(valor):
    if pd.isna(valor):
        return "Não informado"

    mapa = {
        "M": "Masculino",
        "F": "Feminino",
        "I": "Ignorado",
        "1": "Masculino",
        "1.0": "Masculino",
        "2": "Feminino",
        "2.0": "Feminino",
        "9": "Ignorado",
        "9.0": "Ignorado",
    }

    return mapa.get(
        str(valor).strip().upper(),
        "Não informado",
    )


def decodificar_raca(valor):
    mapa = {
        1: "Branca",
        2: "Preta",
        3: "Amarela",
        4: "Parda",
        5: "Indígena",
        9: "Ignorado",
    }

    if pd.isna(valor):
        return "Não informado"

    try:
        return mapa.get(
            int(float(valor)),
            "Não informado",
        )

    except (ValueError, TypeError):
        return "Não informado"


def decodificar_recorrencia(valor):
    mapa = {
        1: "Sim",
        2: "Não",
        9: "Ignorado",
    }

    if pd.isna(valor):
        return "Não informado"

    try:
        return mapa.get(
            int(float(valor)),
            "Não informado",
        )

    except (ValueError, TypeError):
        return "Não informado"


def decodificar_escolaridade_original(valor):
    mapa = {
        0: "Analfabeto",
        1: "1ª a 4ª série incompleta do EF",
        2: "4ª série completa do EF",
        3: "5ª a 8ª série incompleta do EF",
        4: "Ensino fundamental completo",
        5: "Ensino médio incompleto",
        6: "Ensino médio completo",
        7: "Educação superior incompleta",
        8: "Educação superior completa",
        9: "Ignorado",
        10: "Não se aplica",
    }

    if pd.isna(valor):
        return "Não informado"

    try:
        return mapa.get(
            int(float(valor)),
            "Não informado",
        )

    except (ValueError, TypeError):
        return "Não informado"


def agrupar_escolaridade(valor):
    if valor in [
        "Analfabeto",
        "1ª a 4ª série incompleta do EF",
    ]:
        return "Analfabeto a 4ª série incompleta do EF"

    return valor


def estilo_figura(
    fig,
    altura=430,
    margem_esquerda=30,
    margem_direita=30,
    margem_superior=85,
    margem_inferior=55,
):
    fig.update_layout(
        height=altura,
        autosize=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(
            l=margem_esquerda,
            r=margem_direita,
            t=margem_superior,
            b=margem_inferior,
            pad=4,
        ),
        font=dict(
            family="Arial",
            size=12,
        ),
        title=dict(
            x=0.02,
            xanchor="left",
            y=0.97,
            yanchor="top",
            font=dict(
                size=16,
            ),
        ),
        hoverlabel=dict(
            font_size=12,
        ),
    )

    fig.update_xaxes(
        automargin=True,
    )

    fig.update_yaxes(
        automargin=True,
    )

    return fig


def titulo_secao(titulo, descricao=None):
    st.markdown(
        f'<div class="titulo-secao">{titulo}</div>',
        unsafe_allow_html=True,
    )

    if descricao:
        st.markdown(
            f'<div class="descricao-secao">{descricao}</div>',
            unsafe_allow_html=True,
        )


def completude_categoria(
    base,
    coluna,
    categorias_validas,
):
    if len(base) == 0:
        return np.nan

    return (
        base[coluna]
        .isin(categorias_validas)
        .mean()
        * 100
    )


def cramer_v(tabela):
    from scipy.stats import chi2_contingency

    chi2, p, _, esperados = chi2_contingency(tabela)

    n = tabela.to_numpy().sum()
    r, k = tabela.shape

    denominador = min(
        k - 1,
        r - 1,
    )

    if n > 0 and denominador > 0:
        v = sqrt(
            chi2
            / (
                n
                * denominador
            )
        )
    else:
        v = np.nan

    return (
        chi2,
        p,
        v,
        esperados,
    )


def criar_metodos_long(base):
    partes = []

    for campo, descricao in METODOS.items():

        temp = base.loc[
            base[campo] == 1
        ].copy()

        if temp.empty:
            continue

        temp["METODO"] = descricao
        partes.append(temp)

    if not partes:
        return pd.DataFrame()

    return pd.concat(
        partes,
        ignore_index=True,
    )


def limpar_filtros():
    chaves = [
        "f_ano",
        "f_regiao",
        "f_municipio",
        "f_sexo",
        "f_faixa",
        "f_idade",
        "f_raca",
        "f_escolaridade",
        "f_recorrencia",
        "f_metodo",
    ]

    for chave in chaves:
        st.session_state[chave] = []


def imagem_base64(caminho):
    if not os.path.exists(caminho):
        return None

    with open(caminho, "rb") as arquivo:
        dados_imagem = base64.b64encode(
            arquivo.read()
        ).decode("utf-8")

    extensao = os.path.splitext(caminho)[1].lower()

    if extensao == ".png":
        mime = "image/png"
    elif extensao in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    else:
        mime = "image/png"

    return f"data:{mime};base64,{dados_imagem}"


# ============================================================
# 4. ARQUIVOS
# ============================================================

try:

    ARQUIVO_SINAN = localizar_arquivo([
        "BD_AUTOLESÃO_COMPLETO(1).xlsx",
        "BD_AUTOLESÃO_COMPLETO (1).xlsx",
    ])

    ARQUIVO_POP = localizar_arquivo([
        "dados_pernambuco_combinados (1).xlsx",
        "dados_pernambuco_combinados(1).xlsx",
    ])

    ARQUIVO_MALHA = localizar_arquivo([
        "PE_Municipios_2025.zip",
    ])

except FileNotFoundError as erro:

    st.error(str(erro))
    st.stop()


# ============================================================
# 5. CARREGAMENTO
# ============================================================

@st.cache_data(show_spinner=False)
def carregar_dados():

    sinan = pd.read_excel(
        ARQUIVO_SINAN,
        sheet_name="AUTOLESAO_PE",
        engine="openpyxl",
    )

    sinan["IDADE_ANOS"] = (
        sinan["NU_IDADE_N"]
        .apply(idade_em_anos)
    )

    sinan["NU_ANO"] = pd.to_numeric(
        sinan["NU_ANO"],
        errors="coerce",
    )

    sinan["LES_AUTOP"] = pd.to_numeric(
        sinan["LES_AUTOP"],
        errors="coerce",
    )

    sinan = sinan.loc[
        sinan["NU_ANO"].between(
            2014,
            2024,
        )
        & sinan["IDADE_ANOS"].between(
            10,
            19,
        )
        & (
            sinan["LES_AUTOP"] == 1
        )
    ].copy()

    if len(sinan) != 12713:
        raise ValueError(
            "O universo validado possui "
            "12.713 notificações, "
            f"mas foram encontradas {len(sinan)}."
        )

    sinan["NU_ANO"] = (
        sinan["NU_ANO"]
        .astype(int)
    )

    sinan["IDADE_ANOS"] = (
        sinan["IDADE_ANOS"]
        .astype(int)
    )

    sinan["FAIXA_ETARIA"] = pd.cut(
        sinan["IDADE_ANOS"],
        bins=[
            9,
            14,
            19,
        ],
        labels=[
            "10 a 14 anos",
            "15 a 19 anos",
        ],
    ).astype(str)

    sinan["SEXO_DESC"] = (
        sinan["CS_SEXO"]
        .apply(decodificar_sexo)
    )

    sinan["RACA_COR_DESC"] = (
        sinan["CS_RACA"]
        .apply(decodificar_raca)
    )

    sinan["RECORRENCIA_DESC"] = (
        sinan["OUT_VEZES"]
        .apply(decodificar_recorrencia)
    )

    coluna_escolaridade = next(
        (
            coluna
            for coluna in [
                "CS_ESCOL_N",
                "CS_ESCOL",
                "ESCOLARIDADE",
            ]
            if coluna in sinan.columns
        ),
        None,
    )

    if coluna_escolaridade:

        sinan["ESCOLARIDADE_ORIGINAL"] = (
            sinan[coluna_escolaridade]
            .apply(
                decodificar_escolaridade_original
            )
        )

        sinan["ESCOLARIDADE_DESC"] = (
            sinan["ESCOLARIDADE_ORIGINAL"]
            .apply(
                agrupar_escolaridade
            )
        )

    else:

        sinan["ESCOLARIDADE_ORIGINAL"] = (
            "Não informado"
        )

        sinan["ESCOLARIDADE_DESC"] = (
            "Não informado"
        )

    sinan["COD_MUN_6"] = (
        sinan["ID_MN_RESI"]
        .apply(
            codigo_municipio_6d
        )
    )

    for campo in METODOS:

        if campo in sinan.columns:

            sinan[campo] = pd.to_numeric(
                sinan[campo],
                errors="coerce",
            )

        else:

            sinan[campo] = np.nan

    sinan["N_METODOS"] = sum(
        (
            sinan[campo] == 1
        ).astype(int)
        for campo in METODOS
    )

    def classe_metodos(n):

        if n == 0:
            return "Sem método informado"

        if n == 1:
            return "1 método"

        if n == 2:
            return "2 métodos"

        return "3 ou mais métodos"

    sinan["CLASSE_N_METODOS"] = (
        sinan["N_METODOS"]
        .apply(
            classe_metodos
        )
    )

    # --------------------------------------------------------
    # POPULAÇÃO
    # --------------------------------------------------------

    pop = pd.read_excel(
        ARQUIVO_POP,
        sheet_name="consolidado",
        dtype=str,
        engine="openpyxl",
    )

    pop["COD_MUN_6"] = (
        pop["COD_MUN"]
        .apply(
            codigo_municipio_6d
        )
    )

    for coluna in [
        "ANO",
        "IDADE",
        "SEXO",
    ]:

        pop[coluna] = pd.to_numeric(
            pop[coluna],
            errors="coerce",
        )

    def converter_populacao(valor):

        if pd.isna(valor):
            return np.nan

        if isinstance(
            valor,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            return float(valor)

        texto = (
            str(valor)
            .strip()
            .replace(
                " ",
                "",
            )
        )

        if "," in texto:

            texto = (
                texto
                .replace(
                    ".",
                    "",
                )
                .replace(
                    ",",
                    ".",
                )
            )

        try:
            return float(texto)

        except ValueError:
            return np.nan

    pop["POPULAÇÃO"] = (
        pop["POPULAÇÃO"]
        .apply(
            converter_populacao
        )
    )

    pop = pop.loc[
        pop["ANO"].between(
            2014,
            2024,
        )
        & pop["IDADE"].between(
            10,
            19,
        )
    ].copy()

    municipios = (
        pop[
            [
                "COD_MUN_6",
                "MUNICÍPIO",
                "REGIÃO DE SAÚDE",
            ]
        ]
        .drop_duplicates(
            "COD_MUN_6"
        )
    )

    sinan = sinan.merge(
        municipios,
        on="COD_MUN_6",
        how="left",
    )

    return (
        sinan,
        pop,
        coluna_escolaridade,
    )


@st.cache_data(show_spinner=False)
def carregar_malha():

    with tempfile.TemporaryDirectory() as pasta:

        with zipfile.ZipFile(
            ARQUIVO_MALHA,
            "r",
        ) as arquivo:

            arquivo.extractall(
                pasta
            )

        arquivos_shp = []

        for raiz, _, arquivos in os.walk(
            pasta
        ):

            for arquivo in arquivos:

                if arquivo.lower().endswith(
                    ".shp"
                ):

                    arquivos_shp.append(
                        os.path.join(
                            raiz,
                            arquivo,
                        )
                    )

        if not arquivos_shp:

            raise FileNotFoundError(
                "Nenhum arquivo .shp "
                "foi encontrado no ZIP."
            )

        geo = gpd.read_file(
            arquivos_shp[0]
        )

    geo["COD_MUN_6"] = (
        geo["CD_MUN"]
        .apply(
            codigo_municipio_6d
        )
    )

    geo = geo.to_crs(
        epsg=4326
    )

    return geo


try:

    with st.spinner(
        "Carregando e validando os bancos..."
    ):

        df, pop, COL_ESCOLARIDADE = (
            carregar_dados()
        )

        geo = carregar_malha()

except Exception as erro:

    st.error(
        "Não foi possível carregar "
        "os bancos do dashboard."
    )

    st.exception(erro)

    st.stop()


# ============================================================
# 6. CABEÇALHO
# ============================================================

logo_upe_b64 = imagem_base64(
    "logo_upe.png"
)

logo_cdia_b64 = imagem_base64(
    "logo_cdia_saude.png"
)


cab_logo_upe, cab_centro, cab_logo_cdia = st.columns(
    [1.05, 8.90, 1.05],
    gap="small",
)


with cab_logo_upe:

    if logo_upe_b64:

        st.markdown(
            (
                '<div class="header-logo-box">'
                f'<img src="{logo_upe_b64}" '
                'alt="Universidade de Pernambuco">'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


with cab_centro:

    html_cabecalho = (
        '<div class="header-central">'

        '<div class="instituicao">'
        'UNIVERSIDADE DE PERNAMBUCO — UPE · CAMPUS CARUARU'
        '</div>'

        '<div class="curso">'
        'PÓS-GRADUAÇÃO EM CIÊNCIA DE DADOS E INTELIGÊNCIA ARTIFICIAL '
        'APLICADA À SAÚDE'
        '</div>'

        '<div class="titulo-principal">'
        'CENÁRIO EPIDEMIOLÓGICO DE AUTOLESÕES EM ADOLESCENTES EM PERNAMBUCO DE 2014–2024'
        '</div>'

        '<div class="subtitulo">'
        'Notificações de lesão autoprovocada em adolescentes de 10 a 19 anos '
        'residentes em Pernambuco coletadas no banco de dados do SINAN e dados '
        'das estimativas populacionais do IBGE, ambos disponibilizados pelo DATASUS'
        '</div>'

        '</div>'
    )

    st.markdown(
        html_cabecalho,
        unsafe_allow_html=True,
    )


with cab_logo_cdia:

    if logo_cdia_b64:

        st.markdown(
            (
                '<div class="header-logo-box">'
                f'<img src="{logo_cdia_b64}" '
                'alt="CDIA Saúde">'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


st.markdown(
    '<div class="header-bottom-space"></div>',
    unsafe_allow_html=True,
)


# ============================================================
# 7. FILTROS
# ============================================================

with st.sidebar:

    st.markdown(
        "## Filtros"
    )

    st.caption(
        "Os filtros atuam sobre os indicadores "
        "e gráficos compatíveis da página."
    )

    anos = st.multiselect(
        "Ano",
        sorted(
            df["NU_ANO"]
            .unique()
        ),
        key="f_ano",
        placeholder="Todos",
    )

    regioes = st.multiselect(
        "Região de Saúde",
        sorted(
            df["REGIÃO DE SAÚDE"]
            .dropna()
            .unique()
        ),
        key="f_regiao",
        placeholder="Todas",
    )

    base_municipios = (
        df.copy()
    )

    if regioes:

        base_municipios = (
            base_municipios.loc[
                base_municipios[
                    "REGIÃO DE SAÚDE"
                ].isin(
                    regioes
                )
            ]
        )

    municipios = st.multiselect(
        "Município",
        sorted(
            base_municipios[
                "MUNICÍPIO"
            ]
            .dropna()
            .unique()
        ),
        key="f_municipio",
        placeholder="Todos",
    )

    sexos = st.multiselect(
        "Sexo",
        SEXO_VALIDO,
        key="f_sexo",
        placeholder="Todos",
    )

    faixas = st.multiselect(
        "Faixa etária",
        [
            "10 a 14 anos",
            "15 a 19 anos",
        ],
        key="f_faixa",
        placeholder="Todas",
    )

    idades = st.multiselect(
        "Idade simples",
        list(
            range(
                10,
                20,
            )
        ),
        key="f_idade",
        placeholder="Todas",
    )

    racas = st.multiselect(
        "Raça/cor",
        RACA_VALIDA,
        key="f_raca",
        placeholder="Todas",
    )

    escolaridades = st.multiselect(
        "Escolaridade",
        ESCOLARIDADE_VALIDA,
        key="f_escolaridade",
        placeholder="Todas",
    )

    recorrencias = st.multiselect(
        "Ocorrência anterior / recorrência",
        RECORRENCIA_VALIDA,
        key="f_recorrencia",
        placeholder="Todas",
    )

    metodos_selecionados = st.multiselect(
        "Método utilizado",
        list(
            METODOS.values()
        ),
        key="f_metodo",
        placeholder="Todos",
    )

    st.button(
        "Limpar filtros",
        use_container_width=True,
        on_click=limpar_filtros,
    )


# ============================================================
# 8. APLICAÇÃO DOS FILTROS
# ============================================================

dados = df.copy()

if anos:

    dados = dados.loc[
        dados["NU_ANO"].isin(
            anos
        )
    ]

if regioes:

    dados = dados.loc[
        dados[
            "REGIÃO DE SAÚDE"
        ].isin(
            regioes
        )
    ]

if municipios:

    dados = dados.loc[
        dados[
            "MUNICÍPIO"
        ].isin(
            municipios
        )
    ]

if sexos:

    dados = dados.loc[
        dados[
            "SEXO_DESC"
        ].isin(
            sexos
        )
    ]

if faixas:

    dados = dados.loc[
        dados[
            "FAIXA_ETARIA"
        ].isin(
            faixas
        )
    ]

if idades:

    dados = dados.loc[
        dados[
            "IDADE_ANOS"
        ].isin(
            idades
        )
    ]

if racas:

    dados = dados.loc[
        dados[
            "RACA_COR_DESC"
        ].isin(
            racas
        )
    ]

if escolaridades:

    dados = dados.loc[
        dados[
            "ESCOLARIDADE_DESC"
        ].isin(
            escolaridades
        )
    ]

if recorrencias:

    dados = dados.loc[
        dados[
            "RECORRENCIA_DESC"
        ].isin(
            recorrencias
        )
    ]

if metodos_selecionados:

    campos = [
        campo
        for campo, descricao
        in METODOS.items()
        if descricao
        in metodos_selecionados
    ]

    mascara = pd.Series(
        False,
        index=dados.index,
    )

    for campo in campos:

        mascara = (
            mascara
            | (
                dados[campo] == 1
            )
        )

    dados = dados.loc[
        mascara
    ]


# ============================================================
# 9. POPULAÇÃO COMPATÍVEL
# ============================================================

pop_f = pop.copy()

if anos:

    pop_f = pop_f.loc[
        pop_f[
            "ANO"
        ].isin(
            anos
        )
    ]

if municipios:

    codigos = (
        df.loc[
            df[
                "MUNICÍPIO"
            ].isin(
                municipios
            ),
            "COD_MUN_6",
        ]
        .dropna()
        .unique()
    )

    pop_f = pop_f.loc[
        pop_f[
            "COD_MUN_6"
        ].isin(
            codigos
        )
    ]

elif regioes:

    codigos = (
        df.loc[
            df[
                "REGIÃO DE SAÚDE"
            ].isin(
                regioes
            ),
            "COD_MUN_6",
        ]
        .dropna()
        .unique()
    )

    pop_f = pop_f.loc[
        pop_f[
            "COD_MUN_6"
        ].isin(
            codigos
        )
    ]

if sexos:

    codigos_sexo = []

    if "Masculino" in sexos:
        codigos_sexo.append(1)

    if "Feminino" in sexos:
        codigos_sexo.append(2)

    pop_f = pop_f.loc[
        pop_f[
            "SEXO"
        ].isin(
            codigos_sexo
        )
    ]

if idades:

    pop_f = pop_f.loc[
        pop_f[
            "IDADE"
        ].isin(
            idades
        )
    ]

elif faixas:

    idades_pop = []

    if "10 a 14 anos" in faixas:

        idades_pop.extend(
            range(
                10,
                15,
            )
        )

    if "15 a 19 anos" in faixas:

        idades_pop.extend(
            range(
                15,
                20,
            )
        )

    pop_f = pop_f.loc[
        pop_f[
            "IDADE"
        ].isin(
            idades_pop
        )
    ]


filtro_sem_denominador = any([
    bool(racas),
    bool(escolaridades),
    bool(recorrencias),
    bool(metodos_selecionados),
])


# ============================================================
# 10. BASES ANALÍTICAS
# ============================================================

dados_sexo_valido = (
    dados.loc[
        dados[
            "SEXO_DESC"
        ].isin(
            SEXO_VALIDO
        )
    ]
    .copy()
)

dados_raca_valida = (
    dados.loc[
        dados[
            "RACA_COR_DESC"
        ].isin(
            RACA_VALIDA
        )
    ]
    .copy()
)

dados_rec_valida = (
    dados.loc[
        dados[
            "RECORRENCIA_DESC"
        ].isin(
            RECORRENCIA_VALIDA
        )
    ]
    .copy()
)

dados_esc_valida = (
    dados.loc[
        dados[
            "ESCOLARIDADE_DESC"
        ].isin(
            ESCOLARIDADE_VALIDA
        )
    ]
    .copy()
)

dados_metodo_valido = (
    dados.loc[
        dados[
            "N_METODOS"
        ] >= 1
    ]
    .copy()
)

metodos_long = (
    criar_metodos_long(
        dados_metodo_valido
    )
)


# ============================================================
# 11. VISÃO GERAL
# ============================================================

titulo_secao(
    "Visão geral",
    "Síntese epidemiológica da seleção atual.",
)

total = len(dados)

if not filtro_sem_denominador:

    populacao_total = (
        pop_f[
            "POPULAÇÃO"
        ]
        .sum()
    )

    taxa_total = (
        total
        / populacao_total
        * 100000
        if populacao_total > 0
        else np.nan
    )

else:

    taxa_total = np.nan


pct_fem = (
    dados_sexo_valido[
        "SEXO_DESC"
    ]
    .eq(
        "Feminino"
    )
    .mean()
    * 100
    if len(
        dados_sexo_valido
    )
    else np.nan
)


pct_15_19 = (
    dados[
        "FAIXA_ETARIA"
    ]
    .eq(
        "15 a 19 anos"
    )
    .mean()
    * 100
    if len(dados)
    else np.nan
)


pct_rec = (
    dados_rec_valida[
        "RECORRENCIA_DESC"
    ]
    .eq(
        "Sim"
    )
    .mean()
    * 100
    if len(
        dados_rec_valida
    )
    else np.nan
)


k1, k2, k3, k4, k5 = (
    st.columns(5)
)

k1.metric(
    "Notificações",
    numero_br(total),
)

k2.metric(
    "Taxa/100 mil adolescentes",
    numero_br(
        taxa_total,
        1,
    ),
)

k3.metric(
    "Sexo feminino*",
    percentual_br(
        pct_fem
    ),
)

k4.metric(
    "15 a 19 anos",
    percentual_br(
        pct_15_19
    ),
)

k5.metric(
    "Recorrência*",
    percentual_br(
        pct_rec
    ),
)

st.caption(
    "* Percentuais de sexo e recorrência calculados somente "
    "entre registros válidos da respectiva variável."
)

if filtro_sem_denominador:

    st.info(
        "A taxa populacional não é apresentada para esta seleção "
        "porque há filtro de raça/cor, escolaridade, recorrência "
        "ou método, dimensões sem denominadores correspondentes "
        "na base populacional."
    )


# ============================================================
# 12. TENDÊNCIA
# ============================================================

titulo_secao(
    "Tendência",
    "Evolução temporal das notificações e distribuição "
    "segundo raça/cor ao longo do período.",
)

serie = (
    dados
    .groupby(
        "NU_ANO"
    )
    .size()
    .reindex(
        range(
            2014,
            2025,
        ),
        fill_value=0,
    )
    .rename(
        "Notificações"
    )
    .reset_index()
)


if not filtro_sem_denominador:

    pop_ano = (
        pop_f
        .groupby(
            "ANO",
            as_index=False,
        )[
            "POPULAÇÃO"
        ]
        .sum()
    )

    temporal = (
        serie
        .merge(
            pop_ano,
            left_on="NU_ANO",
            right_on="ANO",
            how="left",
        )
    )

    temporal["Taxa"] = np.where(
        temporal[
            "POPULAÇÃO"
        ] > 0,
        temporal[
            "Notificações"
        ]
        / temporal[
            "POPULAÇÃO"
        ]
        * 100000,
        np.nan,
    )

    fig_tempo = make_subplots(
        specs=[
            [
                {
                    "secondary_y":
                    True
                }
            ]
        ]
    )

    fig_tempo.add_trace(
        go.Bar(
            x=temporal[
                "NU_ANO"
            ],
            y=temporal[
                "Notificações"
            ],
            name="Notificações",
            marker_color=AZUL2,
            opacity=0.78,
        ),
        secondary_y=False,
    )

    fig_tempo.add_trace(
        go.Scatter(
            x=temporal[
                "NU_ANO"
            ],
            y=temporal[
                "Taxa"
            ],
            name="Taxa/100 mil",
            mode="lines+markers",
            line=dict(
                color=VERMELHO,
                width=3,
            ),
            marker=dict(
                size=7,
            ),
        ),
        secondary_y=True,
    )

    fig_tempo.update_yaxes(
        title_text="Número de notificações",
        secondary_y=False,
    )

    fig_tempo.update_yaxes(
        title_text="Taxa por 100 mil",
        secondary_y=True,
    )

    fig_tempo.update_layout(
        title=(
            "Evolução das notificações e da taxa "
            "por 100 mil adolescentes"
        ),
        xaxis_title="Ano",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

else:

    fig_tempo = go.Figure(
        go.Bar(
            x=serie[
                "NU_ANO"
            ],
            y=serie[
                "Notificações"
            ],
            marker_color=AZUL2,
        )
    )

    fig_tempo.update_layout(
        title=(
            "Evolução das notificações "
            "na seleção atual"
        ),
        xaxis_title="Ano",
        yaxis_title="Número de notificações",
        showlegend=False,
    )


estilo_figura(
    fig_tempo,
    altura=460,
    margem_esquerda=65,
    margem_direita=65,
    margem_superior=105,
    margem_inferior=65,
)


raca_ano = (
    dados_raca_valida
    .groupby(
        [
            "NU_ANO",
            "RACA_COR_DESC",
        ]
    )
    .size()
    .reset_index(
        name="Notificações"
    )
)


fig_raca_tempo = None

if not raca_ano.empty:

    fig_raca_tempo = px.line(
        raca_ano,
        x="NU_ANO",
        y="Notificações",
        color="RACA_COR_DESC",
        markers=True,
        category_orders={
            "RACA_COR_DESC":
            RACA_VALIDA
        },
        color_discrete_map=(
            CORES_RACA
        ),
        title=(
            "Notificações segundo raça/cor "
            "e ano de notificação"
        ),
    )

    fig_raca_tempo.update_layout(
        xaxis_title="Ano",
        yaxis_title="Número de notificações",
        legend_title="Raça/cor",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    estilo_figura(
        fig_raca_tempo,
        altura=460,
        margem_esquerda=65,
        margem_direita=35,
        margem_superior=105,
        margem_inferior=65,
    )


tend1, tend2 = st.columns(
    2,
    gap="medium",
)

with tend1:

    st.plotly_chart(
        fig_tempo,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )


with tend2:

    if fig_raca_tempo is not None:

        st.plotly_chart(
            fig_raca_tempo,
            use_container_width=True,
            config={
                "responsive": True,
                "displaylogo": False,
            },
        )

        st.caption(
            "Somente registros com raça/cor válida "
            "são representados neste gráfico."
        )

    else:

        st.info(
            "Não há registros válidos de raça/cor "
            "para a seleção atual."
        )


# ============================================================
# 13. TERRITÓRIO
# ============================================================

titulo_secao(
    "Território",
    "Distribuição espacial das notificações entre "
    "municípios e Regiões de Saúde.",
)


pode_taxa_territorial = (
    not filtro_sem_denominador
)


if pode_taxa_territorial:

    medida_territorio = st.radio(
        "Indicador territorial",
        [
            "Taxa por 100 mil",
            "Número de notificações",
        ],
        horizontal=True,
        key="medida_territorio",
    )

else:

    medida_territorio = (
        "Número de notificações"
    )

    st.info(
        "O território está sendo apresentado em números absolutos "
        "porque a seleção contém dimensão sem denominador "
        "populacional correspondente."
    )


municipal = (
    dados
    .dropna(
        subset=[
            "COD_MUN_6"
        ]
    )
    .groupby(
        "COD_MUN_6",
        as_index=False,
    )
    .size()
    .rename(
        columns={
            "size":
            "Notificações"
        }
    )
)


pop_municipal = (
    pop_f
    .groupby(
        "COD_MUN_6",
        as_index=False,
    )[
        "POPULAÇÃO"
    ]
    .sum()
)


nomes_municipios = (
    pop[
        [
            "COD_MUN_6",
            "MUNICÍPIO",
            "REGIÃO DE SAÚDE",
        ]
    ]
    .drop_duplicates(
        "COD_MUN_6"
    )
)


municipal_completo = (
    nomes_municipios
    .merge(
        pop_municipal,
        on="COD_MUN_6",
        how="inner",
    )
    .merge(
        municipal,
        on="COD_MUN_6",
        how="left",
    )
)


municipal_completo[
    "Notificações"
] = (
    municipal_completo[
        "Notificações"
    ]
    .fillna(0)
    .astype(int)
)


municipal_completo[
    "Taxa por 100 mil"
] = np.where(
    municipal_completo[
        "POPULAÇÃO"
    ] > 0,
    municipal_completo[
        "Notificações"
    ]
    / municipal_completo[
        "POPULAÇÃO"
    ]
    * 100000,
    np.nan,
)


mapa = geo.merge(
    municipal_completo,
    on="COD_MUN_6",
    how="left",
)


mapa[
    "Notificações"
] = (
    mapa[
        "Notificações"
    ]
    .fillna(0)
)


coluna_mapa = (
    "Taxa por 100 mil"
    if medida_territorio
    == "Taxa por 100 mil"
    else
    "Notificações"
)


hover = {
    "COD_MUN_6":
    False,
    "Notificações":
    True,
}


if pode_taxa_territorial:

    hover[
        "Taxa por 100 mil"
    ] = ":.1f"


fig_mapa = px.choropleth(
    mapa,
    geojson=(
        mapa
        .__geo_interface__
    ),
    locations="COD_MUN_6",
    featureidkey=(
        "properties.COD_MUN_6"
    ),
    color=coluna_mapa,
    hover_name="NM_MUN",
    hover_data=hover,
    color_continuous_scale=(
        ESCALA_MAGNITUDE
    ),
    title=(
        "Taxa de notificações por município"
        if coluna_mapa
        == "Taxa por 100 mil"
        else
        "Número de notificações por município"
    ),
)


fig_mapa.update_geos(
    fitbounds="locations",
    visible=False,
)


estilo_figura(
    fig_mapa,
    altura=570,
    margem_esquerda=20,
    margem_direita=20,
    margem_superior=80,
    margem_inferior=30,
)


if (
    coluna_mapa
    == "Taxa por 100 mil"
):

    ranking = (
        municipal_completo
        .loc[
            municipal_completo[
                "Notificações"
            ] > 0
        ]
        .sort_values(
            "Taxa por 100 mil",
            ascending=False,
        )
        .head(15)
        .sort_values(
            "Taxa por 100 mil",
            ascending=True,
        )
    )

    fig_rank = px.bar(
        ranking,
        x="Taxa por 100 mil",
        y="MUNICÍPIO",
        orientation="h",
        color="Taxa por 100 mil",
        color_continuous_scale=(
            ESCALA_MAGNITUDE
        ),
        text="Taxa por 100 mil",
        title=(
            "15 maiores taxas municipais"
        ),
    )

    fig_rank.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        cliponaxis=False,
    )

    fig_rank.update_layout(
        xaxis_title="Taxa por 100 mil",
        yaxis_title="",
        coloraxis_showscale=False,
    )

else:

    ranking = (
        municipal_completo
        .sort_values(
            "Notificações",
            ascending=False,
        )
        .head(15)
        .sort_values(
            "Notificações",
            ascending=True,
        )
    )

    fig_rank = px.bar(
        ranking,
        x="Notificações",
        y="MUNICÍPIO",
        orientation="h",
        color="Notificações",
        color_continuous_scale=(
            ESCALA_MAGNITUDE
        ),
        text="Notificações",
        title=(
            "15 maiores números de notificações"
        ),
    )

    fig_rank.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig_rank.update_layout(
        xaxis_title="Número de notificações",
        yaxis_title="",
        coloraxis_showscale=False,
    )


fig_rank.update_yaxes(
    automargin=True,
)


estilo_figura(
    fig_rank,
    altura=570,
    margem_esquerda=120,
    margem_direita=70,
    margem_superior=80,
    margem_inferior=65,
)


terr1, terr2 = st.columns(
    2,
    gap="medium",
)


with terr1:

    st.plotly_chart(
        fig_mapa,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )


with terr2:

    st.plotly_chart(
        fig_rank,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )


regiao = (
    dados
    .groupby(
        "REGIÃO DE SAÚDE"
    )
    .size()
    .reset_index(
        name="Notificações"
    )
    .sort_values(
        "Notificações",
        ascending=True,
    )
)


if not regiao.empty:

    fig_regiao = px.bar(
        regiao,
        x="Notificações",
        y="REGIÃO DE SAÚDE",
        orientation="h",
        color="Notificações",
        color_continuous_scale=(
            ESCALA_MAGNITUDE
        ),
        text="Notificações",
        title=(
            "Notificações segundo Região de Saúde"
        ),
    )

    fig_regiao.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig_regiao.update_layout(
        xaxis_title="Número de notificações",
        yaxis_title="",
        coloraxis_showscale=False,
    )

    estilo_figura(
        fig_regiao,
        altura=480,
        margem_esquerda=120,
        margem_direita=70,
        margem_superior=80,
        margem_inferior=65,
    )

    st.plotly_chart(
        fig_regiao,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )


# ============================================================
# 14. PERFIL EPIDEMIOLÓGICO
# ============================================================

titulo_secao(
    "Perfil epidemiológico",
    "Distribuição das notificações segundo características "
    "sociodemográficas. As análises específicas utilizam "
    "somente respostas analiticamente válidas.",
)


perfil1, perfil2 = st.columns(
    2,
    gap="medium",
)


with perfil1:

    sexo_plot = (
        dados_sexo_valido[
            "SEXO_DESC"
        ]
        .value_counts()
        .reindex(
            SEXO_VALIDO,
            fill_value=0,
        )
        .rename_axis(
            "Sexo"
        )
        .reset_index(
            name="Notificações"
        )
    )

    total_sexo_valido = (
        sexo_plot[
            "Notificações"
        ]
        .sum()
    )

    fig_sexo = px.pie(
        sexo_plot,
        names="Sexo",
        values="Notificações",
        hole=0.50,
        color="Sexo",
        color_discrete_map={
            "Feminino":
            FEMININO,
            "Masculino":
            MASCULINO,
        },
        title=(
            "Notificações segundo sexo"
        ),
    )

    fig_sexo.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Notificações: %{value:,.0f}<br>"
            "Percentual: %{percent}"
            "<extra></extra>"
        ),
    )

    fig_sexo.update_layout(
        showlegend=False,
    )

    estilo_figura(
        fig_sexo,
        altura=420,
        margem_esquerda=30,
        margem_direita=30,
        margem_superior=75,
        margem_inferior=35,
    )

    st.plotly_chart(
        fig_sexo,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    st.caption(
        f"N válido = "
        f"{numero_br(total_sexo_valido)}."
    )


with perfil2:

    faixa_plot = (
        dados[
            "FAIXA_ETARIA"
        ]
        .value_counts()
        .reindex(
            [
                "10 a 14 anos",
                "15 a 19 anos",
            ],
            fill_value=0,
        )
        .rename_axis(
            "Faixa etária"
        )
        .reset_index(
            name="Notificações"
        )
    )

    fig_faixa = px.pie(
        faixa_plot,
        names="Faixa etária",
        values="Notificações",
        hole=0.50,
        color="Faixa etária",
        color_discrete_map={
            "10 a 14 anos":
            IDADE_10_14,
            "15 a 19 anos":
            IDADE_15_19,
        },
        title=(
            "Notificações segundo faixa etária"
        ),
    )

    fig_faixa.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Notificações: %{value:,.0f}<br>"
            "Percentual: %{percent}"
            "<extra></extra>"
        ),
    )

    fig_faixa.update_layout(
        showlegend=False,
    )

    estilo_figura(
        fig_faixa,
        altura=420,
        margem_esquerda=30,
        margem_direita=30,
        margem_superior=75,
        margem_inferior=35,
    )

    st.plotly_chart(
        fig_faixa,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    st.caption(
        f"N = "
        f"{numero_br(len(dados))}."
    )


perfil3, perfil4 = st.columns(
    2,
    gap="medium",
)


with perfil3:

    raca_plot = (
        dados_raca_valida[
            "RACA_COR_DESC"
        ]
        .value_counts()
        .rename_axis(
            "Raça/cor"
        )
        .reset_index(
            name="Notificações"
        )
        .sort_values(
            "Notificações",
            ascending=False,
        )
    )

    total_raca_valida = (
        raca_plot[
            "Notificações"
        ]
        .sum()
    )

    raca_plot[
        "Percentual"
    ] = np.where(
        total_raca_valida > 0,
        raca_plot[
            "Notificações"
        ]
        / total_raca_valida
        * 100,
        0,
    )

    ordem_raca = (
        raca_plot[
            "Raça/cor"
        ]
        .tolist()
    )

    fig_raca = px.bar(
        raca_plot,
        x="Notificações",
        y="Raça/cor",
        orientation="h",
        color="Raça/cor",
        color_discrete_map=(
            CORES_RACA
        ),
        text="Notificações",
        hover_data={
            "Percentual":
            ":.1f"
        },
        title=(
            "Notificações segundo raça/cor"
        ),
    )

    fig_raca.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig_raca.update_yaxes(
        categoryorder="array",
        categoryarray=(
            ordem_raca[::-1]
        ),
        automargin=True,
    )

    fig_raca.update_layout(
        xaxis_title="Número de notificações",
        yaxis_title="",
        showlegend=False,
    )

    estilo_figura(
        fig_raca,
        altura=470,
        margem_esquerda=95,
        margem_direita=70,
        margem_superior=80,
        margem_inferior=65,
    )

    st.plotly_chart(
        fig_raca,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    st.caption(
        f"N válido = "
        f"{numero_br(total_raca_valida)}."
    )


with perfil4:

    esc_plot = (
        dados_esc_valida[
            "ESCOLARIDADE_DESC"
        ]
        .value_counts()
        .rename_axis(
            "Escolaridade"
        )
        .reset_index(
            name="Notificações"
        )
        .sort_values(
            "Notificações",
            ascending=False,
        )
    )

    total_esc_valida = (
        esc_plot[
            "Notificações"
        ]
        .sum()
    )

    esc_plot[
        "Percentual"
    ] = np.where(
        total_esc_valida > 0,
        esc_plot[
            "Notificações"
        ]
        / total_esc_valida
        * 100,
        0,
    )

    ordem_escolaridade = (
        esc_plot[
            "Escolaridade"
        ]
        .tolist()
    )

    fig_esc = px.bar(
        esc_plot,
        x="Notificações",
        y="Escolaridade",
        orientation="h",
        color="Notificações",
        color_continuous_scale=[
            "#A8C4E3",
            "#3E6FA8",
            "#173B6C",
        ],
        text="Notificações",
        hover_data={
            "Percentual":
            ":.1f"
        },
        title=(
            "Notificações segundo escolaridade"
        ),
    )

    fig_esc.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig_esc.update_yaxes(
        categoryorder="array",
        categoryarray=(
            ordem_escolaridade[::-1]
        ),
        automargin=True,
    )

    fig_esc.update_layout(
        xaxis_title="Número de notificações",
        yaxis_title="",
        coloraxis_showscale=False,
    )

    estilo_figura(
        fig_esc,
        altura=470,
        margem_esquerda=210,
        margem_direita=70,
        margem_superior=80,
        margem_inferior=65,
    )

    st.plotly_chart(
        fig_esc,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    st.caption(
        f"N válido = "
        f"{numero_br(total_esc_valida)}. "
        "Analfabeto e 1ª a 4ª série incompleta do EF "
        "foram agrupados."
    )


# ============================================================
# 15. CARACTERÍSTICAS DA AUTOLESÃO
# ============================================================

titulo_secao(
    "Características da autolesão",
    "Métodos registrados, ocorrência anterior/recorrência "
    "e número de métodos registrados em uma mesma notificação.",
)


car1, car2, car3 = st.columns(
    [2, 1, 1],
    gap="medium",
)


with car1:

    if not metodos_long.empty:

        dist_metodo = (
            metodos_long[
                "METODO"
            ]
            .value_counts()
            .rename_axis(
                "Método"
            )
            .reset_index(
                name="Marcações"
            )
            .sort_values(
                "Marcações",
                ascending=False,
            )
        )

        ordem_metodos = (
            dist_metodo[
                "Método"
            ]
            .tolist()
        )

        fig_metodo = px.bar(
            dist_metodo,
            x="Marcações",
            y="Método",
            orientation="h",
            color="Marcações",
            color_continuous_scale=(
                ESCALA_MAGNITUDE
            ),
            text="Marcações",
            title=(
                "Métodos utilizados"
            ),
        )

        fig_metodo.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        fig_metodo.update_yaxes(
            categoryorder="array",
            categoryarray=(
                ordem_metodos[::-1]
            ),
            automargin=True,
        )

        fig_metodo.update_layout(
            xaxis_title="Número de marcações",
            yaxis_title="",
            coloraxis_showscale=False,
        )

        estilo_figura(
            fig_metodo,
            altura=470,
            margem_esquerda=185,
            margem_direita=70,
            margem_superior=80,
            margem_inferior=65,
        )

        st.plotly_chart(
            fig_metodo,
            use_container_width=True,
            config={
                "responsive": True,
                "displaylogo": False,
            },
        )

        st.caption(
            "Resposta múltipla: uma notificação "
            "pode registrar mais de um método."
        )

    else:

        st.info(
            "Nenhum método válido foi identificado "
            "na seleção atual."
        )


with car2:

    rec_plot = (
        dados_rec_valida[
            "RECORRENCIA_DESC"
        ]
        .value_counts()
        .reindex(
            RECORRENCIA_VALIDA,
            fill_value=0,
        )
        .rename_axis(
            "Recorrência"
        )
        .reset_index(
            name="Notificações"
        )
    )

    total_rec_valida = (
        rec_plot[
            "Notificações"
        ]
        .sum()
    )

    fig_rec = px.pie(
        rec_plot,
        names="Recorrência",
        values="Notificações",
        hole=0.52,
        color="Recorrência",
        color_discrete_map={
            "Sim":
            REC_SIM,
            "Não":
            REC_NAO,
        },
        title=(
            "Ocorrência anterior<br>/ recorrência"
        ),
    )

    fig_rec.update_traces(
        textposition="inside",
        textinfo="percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Notificações: %{value:,.0f}<br>"
            "Percentual: %{percent}"
            "<extra></extra>"
        ),
    )

    fig_rec.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.03,
            xanchor="center",
            x=0.5,
        ),
    )

    estilo_figura(
        fig_rec,
        altura=470,
        margem_esquerda=10,
        margem_direita=10,
        margem_superior=95,
        margem_inferior=75,
    )

    st.plotly_chart(
        fig_rec,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    st.caption(
        f"N válido = "
        f"{numero_br(total_rec_valida)}."
    )


with car3:

    metodos_n_plot = (
        dados_metodo_valido[
            "CLASSE_N_METODOS"
        ]
        .value_counts()
        .reindex(
            ORDEM_N_METODOS,
            fill_value=0,
        )
        .rename_axis(
            "Quantidade de métodos"
        )
        .reset_index(
            name="Notificações"
        )
    )

    total_metodo_valido = (
        metodos_n_plot[
            "Notificações"
        ]
        .sum()
    )

    fig_n_metodos = px.pie(
        metodos_n_plot,
        names="Quantidade de métodos",
        values="Notificações",
        hole=0.52,
        color="Quantidade de métodos",
        color_discrete_map={
            "1 método":
            AZUL2,
            "2 métodos":
            "#F28E2B",
            "3 ou mais métodos":
            "#A61B29",
        },
        title=(
            "Distribuição das notificações segundo o "
            "número de métodos registrados"
        ),
    )

    fig_n_metodos.update_traces(
        textposition="inside",
        textinfo="percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Notificações: %{value:,.0f}<br>"
            "Percentual: %{percent}"
            "<extra></extra>"
        ),
    )

    fig_n_metodos.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.03,
            xanchor="center",
            x=0.5,
            font=dict(
                size=10,
            ),
        ),
    )

    estilo_figura(
        fig_n_metodos,
        altura=470,
        margem_esquerda=10,
        margem_direita=10,
        margem_superior=115,
        margem_inferior=90,
    )

    st.plotly_chart(
        fig_n_metodos,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    st.caption(
        f"N válido = "
        f"{numero_br(total_metodo_valido)}. "
        "Cada categoria representa quantos métodos foram "
        "registrados em uma mesma notificação."
    )


# ============================================================
# 16. ANÁLISE ESTATÍSTICA
# ============================================================

titulo_secao(
    "Análise estatística",
    "Associação entre características epidemiológicas "
    "por meio do teste qui-quadrado e V de Cramér.",
)


opcoes_associacao = {

    "Sexo": (
        "SEXO_DESC",
        SEXO_VALIDO,
    ),

    "Faixa etária": (
        "FAIXA_ETARIA",
        [
            "10 a 14 anos",
            "15 a 19 anos",
        ],
    ),

    "Raça/cor": (
        "RACA_COR_DESC",
        RACA_VALIDA,
    ),

    "Escolaridade": (
        "ESCOLARIDADE_DESC",
        ESCOLARIDADE_VALIDA,
    ),

    "Recorrência": (
        "RECORRENCIA_DESC",
        RECORRENCIA_VALIDA,
    ),
}


a1, a2 = st.columns(
    2,
    gap="medium",
)


with a1:

    var1_nome = st.selectbox(
        "Primeira variável",
        list(
            opcoes_associacao.keys()
        ),
        index=0,
        key="assoc_var1",
    )


with a2:

    opcoes_var2 = [
        item
        for item
        in opcoes_associacao
        if item != var1_nome
    ]

    indice_rec = (
        opcoes_var2.index(
            "Recorrência"
        )
        if "Recorrência"
        in opcoes_var2
        else 0
    )

    var2_nome = st.selectbox(
        "Segunda variável",
        opcoes_var2,
        index=indice_rec,
        key="assoc_var2",
    )


col1, validos1 = (
    opcoes_associacao[
        var1_nome
    ]
)

col2, validos2 = (
    opcoes_associacao[
        var2_nome
    ]
)


base_assoc = dados.loc[
    dados[
        col1
    ].isin(
        validos1
    )
    & dados[
        col2
    ].isin(
        validos2
    )
].copy()


tabela_assoc = pd.crosstab(
    base_assoc[
        col1
    ],
    base_assoc[
        col2
    ],
)


if (
    tabela_assoc.shape[0] >= 2
    and tabela_assoc.shape[1] >= 2
):

    try:

        (
            chi2,
            pvalor,
            v,
            esperados,
        ) = cramer_v(
            tabela_assoc
        )

        e1, e2, e3, e4 = (
            st.columns(4)
        )

        e1.metric(
            "N válido",
            numero_br(
                len(
                    base_assoc
                )
            ),
        )

        e2.metric(
            "Qui-quadrado (χ²)",
            numero_br(
                chi2,
                2,
            ),
        )

        e3.metric(
            "p-valor",
            (
                "< 0,001"
                if pvalor < 0.001
                else numero_br(
                    pvalor,
                    3,
                )
            ),
        )

        e4.metric(
            "V de Cramér",
            numero_br(
                v,
                3,
            ),
        )

        graf_assoc = (
            base_assoc
            .groupby(
                [
                    col1,
                    col2,
                ]
            )
            .size()
            .reset_index(
                name="N"
            )
        )

        graf_assoc[
            "Percentual"
        ] = (
            graf_assoc
            .groupby(
                col1
            )[
                "N"
            ]
            .transform(
                lambda x:
                x
                / x.sum()
                * 100
            )
        )

        fig_assoc = px.bar(
            graf_assoc,
            x=col1,
            y="Percentual",
            color=col2,
            barmode="stack",
            hover_data={
                "N":
                True,
                "Percentual":
                ":.1f",
            },
            title=(
                f"{var2_nome} segundo "
                f"{var1_nome.lower()}"
            ),
        )

        fig_assoc.update_layout(
            xaxis_title=(
                var1_nome
            ),
            yaxis_title=(
                "Percentual dentro da categoria (%)"
            ),
            legend_title=(
                var2_nome
            ),
        )

        estilo_figura(
            fig_assoc,
            altura=460,
            margem_esquerda=70,
            margem_direita=50,
            margem_superior=90,
            margem_inferior=90,
        )

        st.plotly_chart(
            fig_assoc,
            use_container_width=True,
            config={
                "responsive": True,
                "displaylogo": False,
            },
        )

        if (
            esperados < 5
        ).any():

            st.warning(
                "Há células com frequência esperada "
                "inferior a 5. A aproximação do teste "
                "qui-quadrado pode ser inadequada "
                "para esta tabela."
            )

        st.caption(
            "O teste utiliza somente registros "
            "com respostas válidas simultaneamente "
            "nas duas variáveis selecionadas."
        )

    except ValueError:

        st.warning(
            "Não foi possível calcular o teste "
            "para esta seleção."
        )

else:

    st.warning(
        "A seleção atual não possui categorias "
        "válidas suficientes para o teste."
    )


# ============================================================
# 17. QUALIDADE DOS DADOS
# ============================================================

titulo_secao(
    "Qualidade dos dados",
    "Evolução da completude das principais variáveis "
    "ao longo do período.",
)

# O gráfico geral "Completude das principais variáveis"
# permanece removido.

linhas_qualidade = []


for ano, base_ano in dados.groupby(
    "NU_ANO"
):

    linhas_qualidade.extend([

        {
            "Ano":
            ano,

            "Variável":
            "Sexo",

            "Completude":
            completude_categoria(
                base_ano,
                "SEXO_DESC",
                SEXO_VALIDO,
            ),
        },

        {
            "Ano":
            ano,

            "Variável":
            "Raça/cor",

            "Completude":
            completude_categoria(
                base_ano,
                "RACA_COR_DESC",
                RACA_VALIDA,
            ),
        },

        {
            "Ano":
            ano,

            "Variável":
            "Escolaridade",

            "Completude":
            completude_categoria(
                base_ano,
                "ESCOLARIDADE_DESC",
                ESCOLARIDADE_VALIDA,
            ),
        },

        {
            "Ano":
            ano,

            "Variável":
            "Recorrência",

            "Completude":
            completude_categoria(
                base_ano,
                "RECORRENCIA_DESC",
                RECORRENCIA_VALIDA,
            ),
        },

        {
            "Ano":
            ano,

            "Variável":
            "Método",

            "Completude":
            (
                base_ano[
                    "N_METODOS"
                ]
                .ge(1)
                .mean()
                * 100
                if len(
                    base_ano
                )
                else np.nan
            ),
        },

    ])


comp_ano = pd.DataFrame(
    linhas_qualidade
)


if not comp_ano.empty:

    matriz_comp = (
        comp_ano
        .pivot(
            index="Variável",
            columns="Ano",
            values="Completude",
        )
    )

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=(
                matriz_comp
                .values
            ),
            x=(
                matriz_comp
                .columns
            ),
            y=(
                matriz_comp
                .index
            ),
            colorscale=[
                [
                    0.0,
                    "#E15759",
                ],
                [
                    0.5,
                    "#FDCB6E",
                ],
                [
                    1.0,
                    "#59A14F",
                ],
            ],
            zmin=0,
            zmax=100,
            text=np.round(
                matriz_comp.values,
                1,
            ),
            texttemplate="%{text}%",
            hovertemplate=(
                "Ano: %{x}<br>"
                "Variável: %{y}<br>"
                "Completude: %{z:.1f}%"
                "<extra></extra>"
            ),
            colorbar=dict(
                title="Completude (%)"
            ),
        )
    )

    fig_heat.update_layout(
        title=(
            "Completude segundo variável "
            "e ano de notificação"
        ),
        xaxis_title="Ano",
        yaxis_title="",
    )

    estilo_figura(
        fig_heat,
        altura=410,
        margem_esquerda=110,
        margem_direita=80,
        margem_superior=80,
        margem_inferior=60,
    )

    st.plotly_chart(
        fig_heat,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )


# ============================================================
# 18. NOTAS METODOLÓGICAS
# ============================================================

st.divider()


with st.expander(
    "Notas metodológicas"
):

    st.markdown(
        """
### População do estudo

Notificações de lesão autoprovocada entre adolescentes de
**10 a 19 anos**, residentes em Pernambuco, registradas entre
**2014 e 2024**.

### Fonte

Sistema de Informação de Agravos de Notificação (SINAN) e
estimativas populacionais do Instituto Brasileiro de Geografia
e Estatística (IBGE), disponibilizados pelo DATASUS.

### Universo validado

O banco analítico reproduz **12.713 notificações** antes
da aplicação dos filtros.

### Respostas válidas

Nas análises específicas de sexo, raça/cor, escolaridade
e recorrência, os percentuais e distribuições são calculados
somente entre registros com informação analiticamente válida.

As categorias **Ignorado**, **Não informado** e, quando
aplicável, **Não se aplica**, não são tratadas como categorias
epidemiológicas dessas distribuições.

Os registros permanecem no banco e contribuem para a
avaliação da completude.

### Escolaridade

Para fins de visualização e análise, as categorias
**Analfabeto** e **1ª a 4ª série incompleta do Ensino
Fundamental** foram agrupadas em:

**Analfabeto a 4ª série incompleta do EF**.

As demais categorias válidas foram preservadas.

### Métodos

Método é uma variável de resposta múltipla. Uma mesma
notificação pode possuir mais de um método marcado.

No gráfico referente à distribuição das notificações segundo
o número de métodos registrados, são consideradas apenas
notificações com pelo menos um método informado, agrupadas em:

- **1 método**
- **2 métodos**
- **3 ou mais métodos**

Essas categorias representam a quantidade de métodos
registrados **em uma mesma notificação**.

O gráfico **Métodos utilizados**, por sua vez, mostra quais
métodos foram marcados. Como há possibilidade de resposta
múltipla, a soma das marcações pode superar o número de
notificações.

### Taxas

As taxas são expressas por **100 mil adolescentes**.

A base populacional permite denominadores por ano,
município, sexo e idade.

Não são calculadas taxas específicas por raça/cor,
escolaridade, recorrência ou método, pois a base populacional
utilizada não contém denominadores correspondentes para
essas dimensões.

Quando filtros dessas dimensões estão ativos, a taxa
populacional da seleção é ocultada para evitar o uso de
denominadores incompatíveis.

Quando a seleção compreende vários anos, o numerador é
dividido pela soma das populações anuais correspondentes.

### Associação estatística

O teste qui-quadrado é utilizado para avaliar associação
entre variáveis categóricas.

O **V de Cramér** é apresentado como medida da magnitude
da associação.

Somente registros válidos simultaneamente nas duas
variáveis selecionadas são utilizados no teste.

### Interpretação epidemiológica

Os resultados descrevem **notificações registradas no
sistema de vigilância**.

Portanto, não representam automaticamente a incidência
real de todos os episódios de autolesão ocorridos na
população adolescente.
        """
    )


# ============================================================
# 19. PRÉ-VISUALIZAÇÃO DOS DADOS
# ============================================================

with st.expander(
    "Pré-visualização dos dados"
):

    st.caption(
        "São apresentadas somente variáveis analíticas. "
        "Identificadores desnecessários não são exibidos."
    )

    colunas_preview = [
        "NU_ANO",
        "IDADE_ANOS",
        "FAIXA_ETARIA",
        "SEXO_DESC",
        "RACA_COR_DESC",
        "ESCOLARIDADE_DESC",
        "RECORRENCIA_DESC",
        "MUNICÍPIO",
        "REGIÃO DE SAÚDE",
        "N_METODOS",
    ]

    preview = dados[
        [
            coluna
            for coluna
            in colunas_preview
            if coluna
            in dados.columns
        ]
    ].copy()

    preview = preview.rename(
        columns={
            "NU_ANO":
            "Ano",

            "IDADE_ANOS":
            "Idade",

            "FAIXA_ETARIA":
            "Faixa etária",

            "SEXO_DESC":
            "Sexo",

            "RACA_COR_DESC":
            "Raça/cor",

            "ESCOLARIDADE_DESC":
            "Escolaridade",

            "RECORRENCIA_DESC":
            "Recorrência",

            "MUNICÍPIO":
            "Município",

            "REGIÃO DE SAÚDE":
            "Região de Saúde",

            "N_METODOS":
            "Nº de métodos",
        }
    )

    st.dataframe(
        preview.head(
            500
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Exibindo até 500 registros de "
        f"{numero_br(len(preview))} notificações "
        "na seleção atual."
    )


# ============================================================
# 20. RODAPÉ
# ============================================================

st.divider()

st.caption(
    "Fonte: Sistema de Informação de Agravos de Notificação "
    "(SINAN) e estimativas populacionais do Instituto Brasileiro "
    "de Geografia e Estatística (IBGE), disponibilizados pelo DATASUS. "
    "Elaboração própria."
)

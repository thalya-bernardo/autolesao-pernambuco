# ============================================================
# CENÁRIO EPIDEMIOLÓGICO DE AUTOLESÕES EM ADOLESCENTES
# PERNAMBUCO | SINAN | 10–19 ANOS | 2014–2024
# ============================================================

import os
import re
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
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. CORES
# ============================================================

AZUL = "#173B6C"
AZUL2 = "#3E6FA8"
VERMELHO = "#E52B35"
FEMININO = "#C44E8B"
MASCULINO = "#2878B5"
IDADE_10_14 = "#2A9D8F"
IDADE_15_19 = "#E9A23B"
REC_SIM = "#D95F02"
REC_NAO = "#4C78A8"
CINZA = "#8B95A5"

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
    "Ignorado": CINZA,
    "Não informado": "#BFC5CC",
}

# ============================================================
# 3. ESTILO
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1600px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    .instituicao {
        text-align:center;
        color:#173B6C;
        font-size:13px;
        font-weight:700;
        letter-spacing:.25px;
        margin-bottom:3px;
    }

    .curso {
        text-align:center;
        color:#5d6875;
        font-size:12px;
        margin-bottom:12px;
    }

    .titulo-principal {
        text-align:center;
        font-size:28px;
        line-height:1.2;
        font-weight:800;
        color:#173B6C;
        margin:5px 0 3px 0;
    }

    .subtitulo {
        text-align:center;
        font-size:14px;
        color:#5c6673;
        margin-bottom:18px;
    }

    div[data-testid="stMetric"] {
        background:white;
        border:1px solid #e2e6eb;
        border-radius:12px;
        padding:12px 14px;
        min-height:105px;
        box-shadow:0 1px 3px rgba(0,0,0,.04);
    }

    div[data-testid="stMetricLabel"] {
        font-size:13px;
    }

    div[data-testid="stMetricValue"] {
        font-size:27px;
        font-weight:750;
        color:#173B6C;
    }

    div[data-testid="stTabs"] button {
        font-weight:600;
    }

    .nota-metodo {
        font-size:12px;
        color:#626d78;
    }

    [data-testid="stSidebar"] {
        border-right:1px solid #e4e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 4. ARQUIVOS
# ============================================================

def localizar_arquivo(opcoes):
    for nome in opcoes:
        if os.path.exists(nome):
            return nome
    raise FileNotFoundError(
        "Nenhum destes arquivos foi encontrado: " + ", ".join(opcoes)
    )

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

# ============================================================
# 5. FUNÇÕES AUXILIARES
# ============================================================

def idade_em_anos(valor):
    if pd.isna(valor):
        return np.nan
    try:
        valor = int(float(valor))
    except Exception:
        return np.nan

    unidade = valor // 1000
    numero = valor % 1000

    return numero if unidade == 4 else np.nan


def codigo_municipio_6d(valor):
    if pd.isna(valor):
        return pd.NA
    try:
        return str(int(float(valor))).zfill(6)[:6]
    except Exception:
        texto = re.sub(r"\D", "", str(valor))
        return texto[:6] if len(texto) >= 6 else pd.NA


def numero_br(valor, casas=0):
    if pd.isna(valor):
        return "—"

    if casas == 0:
        return f"{valor:,.0f}".replace(",", ".")

    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def percentual_br(valor, casas=1):
    return "—" if pd.isna(valor) else f"{numero_br(valor, casas)}%"


def decodificar_sexo(valor):
    if pd.isna(valor):
        return "Não informado"

    texto = str(valor).strip().upper()

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
    return mapa.get(texto, "Não informado")


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
        return mapa.get(int(float(valor)), "Não informado")
    except Exception:
        return "Não informado"


def decodificar_recorrencia(valor):
    mapa = {1: "Sim", 2: "Não", 9: "Ignorado"}

    if pd.isna(valor):
        return "Não informado"

    try:
        return mapa.get(int(float(valor)), "Não informado")
    except Exception:
        return "Não informado"


def decodificar_escolaridade(valor):
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
        return mapa.get(int(float(valor)), "Não informado")
    except Exception:
        return "Não informado"


def ic95_taxa(n, populacao):
    if pd.isna(populacao) or populacao <= 0:
        return np.nan, np.nan

    if n == 0:
        return 0.0, 0.0

    erro = 1.96 * sqrt(n)
    inferior_n = max(0, n - erro)
    superior_n = n + erro

    return (
        inferior_n / populacao * 100000,
        superior_n / populacao * 100000,
    )


def razao_taxas(n1, p1, n0, p0):
    if min(n1, n0, p1, p0) <= 0:
        return np.nan, np.nan, np.nan

    t1 = n1 / p1
    t0 = n0 / p0
    rt = t1 / t0

    se_log = sqrt((1 / n1) + (1 / n0))

    li = np.exp(np.log(rt) - 1.96 * se_log)
    ls = np.exp(np.log(rt) + 1.96 * se_log)

    return rt, li, ls


def cramer_v(tabela):
    try:
        from scipy.stats import chi2_contingency

        chi2, p, dof, expected = chi2_contingency(tabela)

        n = tabela.to_numpy().sum()
        r, k = tabela.shape

        denominador = min(k - 1, r - 1)

        if n == 0 or denominador <= 0:
            return chi2, p, np.nan

        v = sqrt(chi2 / (n * denominador))
        return chi2, p, v

    except Exception:
        return np.nan, np.nan, np.nan


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


def estilo_figura(fig, altura=430):
    fig.update_layout(
        height=altura,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=15, r=15, t=55, b=20),
        font=dict(family="Arial", size=12),
        legend=dict(title=""),
        hoverlabel=dict(font_size=12),
    )
    return fig


# ============================================================
# 6. MÉTODOS
# ============================================================

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

# ============================================================
# 7. CARREGAMENTO DOS DADOS
# ============================================================

@st.cache_data(show_spinner=False)
def carregar_dados():

    sinan = pd.read_excel(
        ARQUIVO_SINAN,
        sheet_name="AUTOLESAO_PE",
        engine="openpyxl",
    )

    sinan["IDADE_ANOS"] = sinan["NU_IDADE_N"].apply(idade_em_anos)
    sinan["NU_ANO"] = pd.to_numeric(sinan["NU_ANO"], errors="coerce")
    sinan["LES_AUTOP"] = pd.to_numeric(sinan["LES_AUTOP"], errors="coerce")

    sinan = sinan[
        sinan["NU_ANO"].between(2014, 2024)
        & sinan["IDADE_ANOS"].between(10, 19)
        & (sinan["LES_AUTOP"] == 1)
    ].copy()

    if len(sinan) != 12713:
        raise ValueError(
            f"O universo validado possui 12.713 notificações, "
            f"mas o processamento encontrou {len(sinan)}."
        )

    sinan["NU_ANO"] = sinan["NU_ANO"].astype(int)
    sinan["IDADE_ANOS"] = sinan["IDADE_ANOS"].astype(int)

    sinan["FAIXA_ETARIA"] = pd.cut(
        sinan["IDADE_ANOS"],
        bins=[9, 14, 19],
        labels=["10 a 14 anos", "15 a 19 anos"],
    ).astype(str)

    sinan["SEXO_DESC"] = sinan["CS_SEXO"].apply(decodificar_sexo)
    sinan["RACA_COR_DESC"] = sinan["CS_RACA"].apply(decodificar_raca)
    sinan["RECORRENCIA_DESC"] = sinan["OUT_VEZES"].apply(
        decodificar_recorrencia
    )

    coluna_escolaridade = None
    for candidato in ["CS_ESCOL_N", "CS_ESCOL", "ESCOLARIDADE"]:
        if candidato in sinan.columns:
            coluna_escolaridade = candidato
            break

    if coluna_escolaridade:
        sinan["ESCOLARIDADE_DESC"] = sinan[coluna_escolaridade].apply(
            decodificar_escolaridade
        )
    else:
        sinan["ESCOLARIDADE_DESC"] = "Não informado"

    sinan["COD_MUN_6"] = sinan["ID_MN_RESI"].apply(codigo_municipio_6d)

    for campo in METODOS:
        if campo in sinan.columns:
            sinan[campo] = pd.to_numeric(sinan[campo], errors="coerce")
        else:
            sinan[campo] = np.nan

    sinan["N_METODOS"] = sum(
        (sinan[campo] == 1).astype(int) for campo in METODOS
    )

    sinan["MULTIPLOS_METODOS"] = np.where(
        sinan["N_METODOS"] >= 2,
        "Dois ou mais métodos",
        np.where(sinan["N_METODOS"] == 1, "Um método", "Nenhum método marcado"),
    )

    pop = pd.read_excel(
        ARQUIVO_POP,
        sheet_name="consolidado",
        dtype=str,
        engine="openpyxl",
    )

    pop["COD_MUN"] = (
        pop["COD_MUN"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    pop["COD_MUN_6"] = pop["COD_MUN"].str[:6]
    pop["ANO"] = pd.to_numeric(pop["ANO"], errors="coerce")
    pop["IDADE"] = pd.to_numeric(pop["IDADE"], errors="coerce")
    pop["SEXO"] = pd.to_numeric(pop["SEXO"], errors="coerce")

    pop["POPULAÇÃO"] = (
        pop["POPULAÇÃO"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    pop["POPULAÇÃO"] = pd.to_numeric(pop["POPULAÇÃO"], errors="coerce")

    pop = pop[
        pop["ANO"].between(2014, 2024)
        & pop["IDADE"].between(10, 19)
    ].copy()

    municipios = (
        pop[["COD_MUN_6", "MUNICÍPIO", "REGIÃO DE SAÚDE"]]
        .drop_duplicates("COD_MUN_6")
    )

    sinan = sinan.merge(
        municipios,
        on="COD_MUN_6",
        how="left",
    )

    return sinan, pop, coluna_escolaridade


@st.cache_data(show_spinner=False)
def carregar_malha():

    pasta = tempfile.mkdtemp()

    with zipfile.ZipFile(ARQUIVO_MALHA, "r") as z:
        z.extractall(pasta)

    arquivos_shp = []

    for raiz, _, arquivos in os.walk(pasta):
        for arquivo in arquivos:
            if arquivo.lower().endswith(".shp"):
                arquivos_shp.append(os.path.join(raiz, arquivo))

    if not arquivos_shp:
        raise FileNotFoundError("Nenhum arquivo .shp foi encontrado no ZIP.")

    geo = gpd.read_file(arquivos_shp[0])

    geo["COD_MUN_6"] = geo["CD_MUN"].astype(str).str[:6]
    geo = geo.to_crs(epsg=4326)

    return geo


try:
    with st.spinner("Carregando e validando os bancos..."):
        df, pop, COL_ESCOLARIDADE = carregar_dados()
        geo = carregar_malha()

except Exception as erro:
    st.error("Não foi possível carregar os bancos do dashboard.")
    st.exception(erro)
    st.stop()

# ============================================================
# 8. CABEÇALHO INSTITUCIONAL
# ============================================================

cab1, cab2, cab3 = st.columns([1, 5, 1])

with cab1:
    if os.path.exists("logo_upe.png"):
        st.image("logo_upe.png", width=125)

with cab2:
    st.markdown(
        """
        <div class="instituicao">
        UNIVERSIDADE DE PERNAMBUCO — UPE · CAMPUS CARUARU
        </div>
        <div class="curso">
        PÓS-GRADUAÇÃO EM CIÊNCIA DE DADOS E INTELIGÊNCIA ARTIFICIAL
        APLICADA À SAÚDE · VISUALIZAÇÃO DE DADOS
        </div>
        <div class="titulo-principal">
        CENÁRIO EPIDEMIOLÓGICO DE AUTOLESÕES EM ADOLESCENTES
        EM PERNAMBUCO DE 2014–2024
        </div>
        <div class="subtitulo">
        Notificações de lesão autoprovocada em adolescentes de 10 a 19 anos
        residentes em Pernambuco
        </div>
        """,
        unsafe_allow_html=True,
    )

with cab3:
    if os.path.exists("logo_cdia_saude.png"):
        st.image("logo_cdia_saude.png", width=125)

# ============================================================
# 9. FILTROS LATERAIS
# ============================================================

with st.sidebar:

    st.markdown("## Filtros")
    st.caption("Todos os gráficos compatíveis respondem aos filtros selecionados.")

    anos = st.multiselect(
        "Ano",
        sorted(df["NU_ANO"].unique()),
        key="f_ano",
        placeholder="Todos",
    )

    regioes = st.multiselect(
        "Região de Saúde",
        sorted(df["REGIÃO DE SAÚDE"].dropna().unique()),
        key="f_regiao",
        placeholder="Todas",
    )

    base_municipios = df.copy()

    if regioes:
        base_municipios = base_municipios[
            base_municipios["REGIÃO DE SAÚDE"].isin(regioes)
        ]

    municipios = st.multiselect(
        "Município",
        sorted(base_municipios["MUNICÍPIO"].dropna().unique()),
        key="f_municipio",
        placeholder="Todos",
    )

    sexos = st.multiselect(
        "Sexo",
        ["Feminino", "Masculino", "Ignorado", "Não informado"],
        key="f_sexo",
        placeholder="Todos",
    )

    faixas = st.multiselect(
        "Faixa etária",
        ["10 a 14 anos", "15 a 19 anos"],
        key="f_faixa",
        placeholder="Todas",
    )

    idades = st.multiselect(
        "Idade simples",
        list(range(10, 20)),
        key="f_idade",
        placeholder="Todas",
    )

    racas = st.multiselect(
        "Raça/cor",
        [
            "Parda",
            "Branca",
            "Preta",
            "Amarela",
            "Indígena",
            "Ignorado",
            "Não informado",
        ],
        key="f_raca",
        placeholder="Todas",
    )

    escolaridades = st.multiselect(
        "Escolaridade",
        sorted(df["ESCOLARIDADE_DESC"].dropna().unique()),
        key="f_escolaridade",
        placeholder="Todas",
    )

    recorrencias = st.multiselect(
        "Ocorrência anterior / recorrência",
        ["Sim", "Não", "Ignorado", "Não informado"],
        key="f_recorrencia",
        placeholder="Todas",
    )

    metodos_selecionados = st.multiselect(
        "Método utilizado",
        list(METODOS.values()),
        key="f_metodo",
        placeholder="Todos",
    )

    st.button(
        "Limpar filtros",
        use_container_width=True,
        on_click=limpar_filtros,
    )

# ============================================================
# 10. FILTRAGEM
# ============================================================

dados = df.copy()

if anos:
    dados = dados[dados["NU_ANO"].isin(anos)]

if regioes:
    dados = dados[dados["REGIÃO DE SAÚDE"].isin(regioes)]

if municipios:
    dados = dados[dados["MUNICÍPIO"].isin(municipios)]

if sexos:
    dados = dados[dados["SEXO_DESC"].isin(sexos)]

if faixas:
    dados = dados[dados["FAIXA_ETARIA"].isin(faixas)]

if idades:
    dados = dados[dados["IDADE_ANOS"].isin(idades)]

if racas:
    dados = dados[dados["RACA_COR_DESC"].isin(racas)]

if escolaridades:
    dados = dados[dados["ESCOLARIDADE_DESC"].isin(escolaridades)]

if recorrencias:
    dados = dados[dados["RECORRENCIA_DESC"].isin(recorrencias)]

if metodos_selecionados:

    campos = [
        campo
        for campo, descricao in METODOS.items()
        if descricao in metodos_selecionados
    ]

    mascara = pd.Series(False, index=dados.index)

    for campo in campos:
        mascara = mascara | (dados[campo] == 1)

    dados = dados[mascara]

# ============================================================
# 11. POPULAÇÃO COMPATÍVEL
# ============================================================

pop_f = pop.copy()

if anos:
    pop_f = pop_f[pop_f["ANO"].isin(anos)]

if municipios:

    codigos = df.loc[
        df["MUNICÍPIO"].isin(municipios),
        "COD_MUN_6",
    ].dropna().unique()

    pop_f = pop_f[pop_f["COD_MUN_6"].isin(codigos)]

elif regioes:

    codigos = df.loc[
        df["REGIÃO DE SAÚDE"].isin(regioes),
        "COD_MUN_6",
    ].dropna().unique()

    pop_f = pop_f[pop_f["COD_MUN_6"].isin(codigos)]

if sexos:

    codigos_sexo = []

    if "Masculino" in sexos:
        codigos_sexo.append(1)

    if "Feminino" in sexos:
        codigos_sexo.append(2)

    if codigos_sexo:
        pop_f = pop_f[pop_f["SEXO"].isin(codigos_sexo)]

if idades:
    pop_f = pop_f[pop_f["IDADE"].isin(idades)]

elif faixas:

    idades_pop = []

    if "10 a 14 anos" in faixas:
        idades_pop.extend(range(10, 15))

    if "15 a 19 anos" in faixas:
        idades_pop.extend(range(15, 20))

    pop_f = pop_f[pop_f["IDADE"].isin(idades_pop)]

filtro_sem_denominador = any([
    bool(racas),
    bool(escolaridades),
    bool(recorrencias),
    bool(metodos_selecionados),
    any(s in ["Ignorado", "Não informado"] for s in sexos),
])

# ============================================================
# 12. MÉTODOS EM FORMATO LONGO
# ============================================================

def criar_metodos_long(base):

    partes = []

    for campo, descricao in METODOS.items():

        temp = base[base[campo] == 1].copy()

        if temp.empty:
            continue

        temp["METODO"] = descricao
        partes.append(temp)

    if not partes:
        return pd.DataFrame()

    return pd.concat(partes, ignore_index=True)


metodos_long = criar_metodos_long(dados)

# ============================================================
# 13. ABAS
# ============================================================

abas = st.tabs([
    "Visão geral",
    "Tendência",
    "Território",
    "Perfil",
    "Características",
    "Análise estatística",
    "Qualidade",
])

# ============================================================
# 14. VISÃO GERAL
# ============================================================

with abas[0]:

    total = len(dados)

    if not filtro_sem_denominador:
        populacao_total = pop_f["POPULAÇÃO"].sum()

        taxa_total = (
            total / populacao_total * 100000
            if populacao_total > 0
            else np.nan
        )
    else:
        populacao_total = np.nan
        taxa_total = np.nan

    sexo_valido = dados[
        dados["SEXO_DESC"].isin(["Feminino", "Masculino"])
    ]

    pct_fem = (
        (sexo_valido["SEXO_DESC"] == "Feminino").mean() * 100
        if len(sexo_valido)
        else np.nan
    )

    pct_15_19 = (
        (dados["FAIXA_ETARIA"] == "15 a 19 anos").mean() * 100
        if len(dados)
        else np.nan
    )

    rec_valida = dados[
        dados["RECORRENCIA_DESC"].isin(["Sim", "Não"])
    ]

    pct_rec = (
        (rec_valida["RECORRENCIA_DESC"] == "Sim").mean() * 100
        if len(rec_valida)
        else np.nan
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Notificações", numero_br(total))

    k2.metric(
        "Taxa/100 mil adolescentes",
        numero_br(taxa_total, 1) if pd.notna(taxa_total) else "—",
    )

    k3.metric("Sexo feminino", percentual_br(pct_fem, 1))
    k4.metric("15 a 19 anos", percentual_br(pct_15_19, 1))
    k5.metric("Recorrência*", percentual_br(pct_rec, 1))

    st.caption(
        "* Recorrência calculada entre respostas válidas (Sim/Não). "
        "A taxa populacional só é exibida quando os filtros possuem "
        "denominador populacional compatível."
    )

    if filtro_sem_denominador:
        st.info(
            "A taxa populacional não é recalculada nesta seleção porque "
            "há filtro sem denominador populacional correspondente "
            "(raça/cor, escolaridade, recorrência, método ou sexo sem "
            "categoria populacional compatível)."
        )

    st.markdown("### Evolução temporal")

    serie = (
        dados.groupby("NU_ANO")
        .size()
        .reindex(range(2014, 2025), fill_value=0)
        .rename("Notificações")
        .reset_index()
        .rename(columns={"index": "NU_ANO"})
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=serie["NU_ANO"],
            y=serie["Notificações"],
            name="Notificações",
            marker_color=AZUL2,
            hovertemplate="<b>%{x}</b><br>Notificações: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Número de notificações por ano",
        xaxis_title="Ano",
        yaxis_title="Notificações",
        showlegend=False,
    )

    estilo_figura(fig, 390)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Principais achados da seleção")

    achados = []

    if total > 0:
        achados.append(
            f"Foram identificadas **{numero_br(total)} notificações** "
            f"na seleção atual."
        )

    if pd.notna(pct_fem):
        achados.append(
            f"O sexo feminino corresponde a **{percentual_br(pct_fem, 1)}** "
            f"das notificações com sexo válido."
        )

    if pd.notna(pct_15_19):
        achados.append(
            f"Adolescentes de 15 a 19 anos representam "
            f"**{percentual_br(pct_15_19, 1)}** das notificações selecionadas."
        )

    if pd.notna(pct_rec):
        achados.append(
            f"Entre os registros com informação válida de recorrência, "
            f"**{percentual_br(pct_rec, 1)}** registraram ocorrência anterior."
        )

    for texto in achados:
        st.markdown(f"- {texto}")

# ============================================================
# 15. TENDÊNCIA
# ============================================================

with abas[1]:

    st.subheader("Tendência temporal")

    esquerda, direita = st.columns([1.65, 1])

    with esquerda:

        serie = (
            dados.groupby("NU_ANO")
            .size()
            .reindex(range(2014, 2025), fill_value=0)
            .rename("Notificações")
            .reset_index()
            .rename(columns={"index": "NU_ANO"})
        )

        if not filtro_sem_denominador:

            pop_ano = (
                pop_f.groupby("ANO", as_index=False)["POPULAÇÃO"]
                .sum()
            )

            temporal = serie.merge(
                pop_ano,
                left_on="NU_ANO",
                right_on="ANO",
                how="left",
            )

            temporal["Taxa"] = np.where(
                temporal["POPULAÇÃO"] > 0,
                temporal["Notificações"] / temporal["POPULAÇÃO"] * 100000,
                np.nan,
            )

            fig_tempo = make_subplots(specs=[[{"secondary_y": True}]])

            fig_tempo.add_trace(
                go.Bar(
                    x=temporal["NU_ANO"],
                    y=temporal["Notificações"],
                    name="Notificações",
                    marker_color=AZUL2,
                    opacity=0.78,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Notificações: %{y:,.0f}<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

            fig_tempo.add_trace(
                go.Scatter(
                    x=temporal["NU_ANO"],
                    y=temporal["Taxa"],
                    name="Taxa/100 mil",
                    mode="lines+markers",
                    line=dict(color=VERMELHO, width=3),
                    marker=dict(size=7),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Taxa: %{y:.1f}/100 mil<extra></extra>"
                    ),
                ),
                secondary_y=True,
            )

            fig_tempo.update_yaxes(
                title_text="Número de notificações",
                secondary_y=False,
            )

            fig_tempo.update_yaxes(
                title_text="Taxa por 100 mil adolescentes",
                secondary_y=True,
            )

            fig_tempo.update_layout(
                title="Notificações e taxa por 100 mil adolescentes",
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
                    x=serie["NU_ANO"],
                    y=serie["Notificações"],
                    marker_color=AZUL2,
                    name="Notificações",
                )
            )

            fig_tempo.update_layout(
                title="Número de notificações por ano",
                xaxis_title="Ano",
                yaxis_title="Notificações",
            )

        estilo_figura(fig_tempo, 470)
        st.plotly_chart(fig_tempo, use_container_width=True)

        if filtro_sem_denominador:
            st.caption(
                "A linha de taxa é ocultada porque a seleção contém "
                "dimensão sem denominador populacional correspondente."
            )

    with direita:

        raca_ano = (
            dados.groupby(["NU_ANO", "RACA_COR_DESC"])
            .size()
            .reset_index(name="Notificações")
        )

        ordem_raca = [
            "Parda",
            "Branca",
            "Preta",
            "Amarela",
            "Indígena",
            "Ignorado",
            "Não informado",
        ]

        fig_raca_tempo = px.line(
            raca_ano,
            x="NU_ANO",
            y="Notificações",
            color="RACA_COR_DESC",
            markers=True,
            category_orders={"RACA_COR_DESC": ordem_raca},
            color_discrete_map=CORES_RACA,
            title="Notificações segundo raça/cor e ano",
        )

        fig_raca_tempo.update_layout(
            xaxis_title="Ano",
            yaxis_title="Notificações",
            legend_title="Raça/cor",
        )

        estilo_figura(fig_raca_tempo, 470)
        st.plotly_chart(fig_raca_tempo, use_container_width=True)

        st.caption(
            "Raça/cor é apresentada por número de notificações. "
            "Não há denominador populacional por raça/cor nesta base."
        )

    st.markdown("### Taxas específicas por sexo e faixa etária")

    base_taxas = df.copy()

    if anos:
        base_taxas = base_taxas[base_taxas["NU_ANO"].isin(anos)]

    if regioes:
        base_taxas = base_taxas[base_taxas["REGIÃO DE SAÚDE"].isin(regioes)]

    if municipios:
        base_taxas = base_taxas[base_taxas["MUNICÍPIO"].isin(municipios)]

    grupos = []

    for sexo_nome, sexo_cod in [("Feminino", 2), ("Masculino", 1)]:
        for faixa_nome, faixa_idades in [
            ("10 a 14 anos", range(10, 15)),
            ("15 a 19 anos", range(15, 20)),
        ]:

            eventos = base_taxas[
                (base_taxas["SEXO_DESC"] == sexo_nome)
                & (base_taxas["IDADE_ANOS"].isin(faixa_idades))
            ]

            p = pop.copy()

            if anos:
                p = p[p["ANO"].isin(anos)]

            if municipios:
                cods = df.loc[
                    df["MUNICÍPIO"].isin(municipios), "COD_MUN_6"
                ].unique()
                p = p[p["COD_MUN_6"].isin(cods)]

            elif regioes:
                cods = df.loc[
                    df["REGIÃO DE SAÚDE"].isin(regioes), "COD_MUN_6"
                ].unique()
                p = p[p["COD_MUN_6"].isin(cods)]

            p = p[
                (p["SEXO"] == sexo_cod)
                & (p["IDADE"].isin(faixa_idades))
            ]

            n = len(eventos)
            populacao_grupo = p["POPULAÇÃO"].sum()

            taxa_grupo = (
                n / populacao_grupo * 100000
                if populacao_grupo > 0
                else np.nan
            )

            li, ls = ic95_taxa(n, populacao_grupo)

            grupos.append({
                "Grupo": f"{sexo_nome} · {faixa_nome}",
                "Notificações": n,
                "População": populacao_grupo,
                "Taxa": taxa_grupo,
                "LI95": li,
                "LS95": ls,
            })

    grupos = pd.DataFrame(grupos)

    fig_grupos = go.Figure()

    fig_grupos.add_trace(
        go.Scatter(
            x=grupos["Taxa"],
            y=grupos["Grupo"],
            mode="markers",
            marker=dict(size=10, color=AZUL),
            error_x=dict(
                type="data",
                symmetric=False,
                array=grupos["LS95"] - grupos["Taxa"],
                arrayminus=grupos["Taxa"] - grupos["LI95"],
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Taxa: %{x:.1f}/100 mil<extra></extra>"
            ),
        )
    )

    fig_grupos.update_layout(
        title="Taxas específicas e IC95% por sexo e faixa etária",
        xaxis_title="Taxa por 100 mil adolescentes",
        yaxis_title="",
        showlegend=False,
    )

    estilo_figura(fig_grupos, 360)
    st.plotly_chart(fig_grupos, use_container_width=True)
    # ============================================================
# 16. TERRITÓRIO
# ============================================================

with abas[2]:

    st.subheader("Distribuição territorial")

    pode_taxa_territorial = not filtro_sem_denominador

    if pode_taxa_territorial:
        medida_territorio = st.radio(
            "Indicador do mapa",
            ["Taxa por 100 mil", "Número de notificações"],
            horizontal=True,
            key="medida_territorio",
        )
    else:
        medida_territorio = "Número de notificações"

        st.info(
            "O mapa está exibindo número de notificações porque a seleção "
            "possui dimensão sem denominador populacional correspondente."
        )

    # --------------------------------------------------------
    # Notificações por município
    # --------------------------------------------------------

    municipal = (
        dados.dropna(subset=["COD_MUN_6"])
        .groupby(["COD_MUN_6", "MUNICÍPIO"], as_index=False)
        .size()
        .rename(columns={"size": "Notificações"})
    )

    # Todos os municípios da população filtrada
    pop_municipal = (
        pop_f.groupby("COD_MUN_6", as_index=False)["POPULAÇÃO"]
        .sum()
    )

    nomes_municipios = (
        pop[["COD_MUN_6", "MUNICÍPIO", "REGIÃO DE SAÚDE"]]
        .drop_duplicates("COD_MUN_6")
    )

    municipal_completo = nomes_municipios.merge(
        pop_municipal,
        on="COD_MUN_6",
        how="inner",
    )

    municipal_completo = municipal_completo.merge(
        municipal[["COD_MUN_6", "Notificações"]],
        on="COD_MUN_6",
        how="left",
    )

    municipal_completo["Notificações"] = (
        municipal_completo["Notificações"]
        .fillna(0)
        .astype(int)
    )

    municipal_completo["Taxa por 100 mil"] = np.where(
        municipal_completo["POPULAÇÃO"] > 0,
        municipal_completo["Notificações"]
        / municipal_completo["POPULAÇÃO"]
        * 100000,
        np.nan,
    )

    intervalos = municipal_completo.apply(
        lambda linha: ic95_taxa(
            linha["Notificações"],
            linha["POPULAÇÃO"],
        ),
        axis=1,
    )

    municipal_completo["LI95"] = [x[0] for x in intervalos]
    municipal_completo["LS95"] = [x[1] for x in intervalos]

    mapa = geo.merge(
        municipal_completo,
        on="COD_MUN_6",
        how="left",
    )

    mapa["Notificações"] = mapa["Notificações"].fillna(0)

    coluna_mapa = (
        "Taxa por 100 mil"
        if medida_territorio == "Taxa por 100 mil"
        else "Notificações"
    )

    c1, c2 = st.columns([1.35, 0.85])

    with c1:

        hover = {
            "COD_MUN_6": False,
            "Notificações": True,
        }

        if pode_taxa_territorial:
            hover["Taxa por 100 mil"] = ":.1f"

        fig_mapa = px.choropleth(
            mapa,
            geojson=mapa.__geo_interface__,
            locations="COD_MUN_6",
            featureidkey="properties.COD_MUN_6",
            color=coluna_mapa,
            hover_name="NM_MUN",
            hover_data=hover,
            color_continuous_scale=ESCALA_MAGNITUDE,
            title=(
                "Taxa de notificações por município de residência"
                if coluna_mapa == "Taxa por 100 mil"
                else "Número de notificações por município de residência"
            ),
        )

        fig_mapa.update_geos(
            fitbounds="locations",
            visible=False,
        )

        fig_mapa.update_layout(
            coloraxis_colorbar=dict(
                title=(
                    "Taxa/100 mil"
                    if coluna_mapa == "Taxa por 100 mil"
                    else "Notificações"
                )
            )
        )

        estilo_figura(fig_mapa, 570)
        st.plotly_chart(fig_mapa, use_container_width=True)

    with c2:

        if coluna_mapa == "Taxa por 100 mil":

            ranking = (
                municipal_completo[
                    municipal_completo["Notificações"] > 0
                ]
                .sort_values("Taxa por 100 mil", ascending=False)
                .head(15)
                .sort_values("Taxa por 100 mil")
            )

            fig_rank = px.bar(
                ranking,
                x="Taxa por 100 mil",
                y="MUNICÍPIO",
                orientation="h",
                color="Taxa por 100 mil",
                color_continuous_scale=ESCALA_MAGNITUDE,
                title="15 maiores taxas municipais",
                hover_data={
                    "Notificações": True,
                    "POPULAÇÃO": ":,.0f",
                    "Taxa por 100 mil": ":.1f",
                },
            )

            fig_rank.update_layout(
                xaxis_title="Taxa por 100 mil adolescentes",
                yaxis_title="",
                coloraxis_showscale=False,
            )

        else:

            ranking = (
                municipal_completo
                .sort_values("Notificações", ascending=False)
                .head(15)
                .sort_values("Notificações")
            )

            fig_rank = px.bar(
                ranking,
                x="Notificações",
                y="MUNICÍPIO",
                orientation="h",
                color="Notificações",
                color_continuous_scale=ESCALA_MAGNITUDE,
                title="15 maiores números de notificações",
            )

            fig_rank.update_layout(
                xaxis_title="Número de notificações",
                yaxis_title="",
                coloraxis_showscale=False,
            )

        estilo_figura(fig_rank, 570)
        st.plotly_chart(fig_rank, use_container_width=True)

    # --------------------------------------------------------
    # Carga x magnitude
    # --------------------------------------------------------

    if pode_taxa_territorial:

        st.markdown("### Carga de notificações × magnitude da taxa")

        dispersao = municipal_completo[
            municipal_completo["POPULAÇÃO"] > 0
        ].copy()

        fig_disp = px.scatter(
            dispersao,
            x="Taxa por 100 mil",
            y="Notificações",
            size="POPULAÇÃO",
            hover_name="MUNICÍPIO",
            hover_data={
                "REGIÃO DE SAÚDE": True,
                "POPULAÇÃO": ":,.0f",
                "Taxa por 100 mil": ":.1f",
                "Notificações": True,
            },
            size_max=45,
            title=(
                "Municípios segundo taxa, número de notificações "
                "e população adolescente"
            ),
        )

        fig_disp.update_traces(
            marker=dict(
                color=AZUL2,
                opacity=0.68,
                line=dict(width=0.5, color="white"),
            )
        )

        fig_disp.update_layout(
            xaxis_title="Taxa por 100 mil adolescentes",
            yaxis_title="Número de notificações",
        )

        estilo_figura(fig_disp, 480)
        st.plotly_chart(fig_disp, use_container_width=True)

        st.caption(
            "O tamanho dos pontos representa a população adolescente. "
            "A visualização ajuda a distinguir municípios com grande carga "
            "absoluta daqueles com elevada magnitude relativa."
        )

    # --------------------------------------------------------
    # Regiões de Saúde
    # --------------------------------------------------------

    st.markdown("### Regiões de Saúde")

    regiao = (
        dados.groupby("REGIÃO DE SAÚDE")
        .size()
        .reset_index(name="Notificações")
        .sort_values("Notificações")
    )

    fig_regiao = px.bar(
        regiao,
        x="Notificações",
        y="REGIÃO DE SAÚDE",
        orientation="h",
        color="Notificações",
        color_continuous_scale=ESCALA_MAGNITUDE,
        title="Notificações segundo Região de Saúde",
        text="Notificações",
    )

    fig_regiao.update_layout(
        xaxis_title="Número de notificações",
        yaxis_title="",
        coloraxis_showscale=False,
    )

    estilo_figura(fig_regiao, 460)
    st.plotly_chart(fig_regiao, use_container_width=True)

# ============================================================
# 17. PERFIL
# ============================================================

with abas[3]:

    st.subheader("Perfil epidemiológico das notificações")

    p1, p2 = st.columns(2)

    # --------------------------------------------------------
    # Sexo
    # --------------------------------------------------------

    with p1:

        sexo_plot = (
            dados["SEXO_DESC"]
            .value_counts(dropna=False)
            .rename_axis("Sexo")
            .reset_index(name="Notificações")
        )

        sexo_plot["Percentual"] = (
            sexo_plot["Notificações"]
            / sexo_plot["Notificações"].sum()
            * 100
            if sexo_plot["Notificações"].sum() > 0
            else 0
        )

        fig_sexo = px.bar(
            sexo_plot,
            x="Sexo",
            y="Notificações",
            color="Sexo",
            color_discrete_map={
                "Feminino": FEMININO,
                "Masculino": MASCULINO,
                "Ignorado": CINZA,
                "Não informado": "#BFC5CC",
            },
            text="Notificações",
            title="Notificações segundo sexo",
            hover_data={"Percentual": ":.1f"},
        )

        fig_sexo.update_layout(
            xaxis_title="",
            yaxis_title="Número de notificações",
            showlegend=False,
        )

        estilo_figura(fig_sexo, 400)
        st.plotly_chart(fig_sexo, use_container_width=True)

    # --------------------------------------------------------
    # Faixa etária
    # --------------------------------------------------------

    with p2:

        faixa_plot = (
            dados["FAIXA_ETARIA"]
            .value_counts()
            .reindex(
                ["10 a 14 anos", "15 a 19 anos"],
                fill_value=0,
            )
            .rename_axis("Faixa etária")
            .reset_index(name="Notificações")
        )

        faixa_plot["Percentual"] = np.where(
            faixa_plot["Notificações"].sum() > 0,
            faixa_plot["Notificações"]
            / faixa_plot["Notificações"].sum()
            * 100,
            0,
        )

        fig_faixa = px.bar(
            faixa_plot,
            x="Faixa etária",
            y="Notificações",
            color="Faixa etária",
            color_discrete_map={
                "10 a 14 anos": IDADE_10_14,
                "15 a 19 anos": IDADE_15_19,
            },
            text="Notificações",
            title="Notificações segundo faixa etária",
            hover_data={"Percentual": ":.1f"},
        )

        fig_faixa.update_layout(
            xaxis_title="",
            yaxis_title="Número de notificações",
            showlegend=False,
        )

        estilo_figura(fig_faixa, 400)
        st.plotly_chart(fig_faixa, use_container_width=True)

    # --------------------------------------------------------
    # Idade simples
    # --------------------------------------------------------

    idade_plot = (
        dados["IDADE_ANOS"]
        .value_counts()
        .reindex(range(10, 20), fill_value=0)
        .rename_axis("Idade")
        .reset_index(name="Notificações")
    )

    fig_idade = px.bar(
        idade_plot,
        x="Idade",
        y="Notificações",
        color="Notificações",
        color_continuous_scale=[
            "#D7F0ED",
            "#2A9D8F",
            "#E9A23B",
        ],
        text="Notificações",
        title="Distribuição das notificações por idade simples",
    )

    fig_idade.update_layout(
        xaxis=dict(dtick=1),
        xaxis_title="Idade (anos)",
        yaxis_title="Número de notificações",
        coloraxis_showscale=False,
    )

    estilo_figura(fig_idade, 410)
    st.plotly_chart(fig_idade, use_container_width=True)

    # --------------------------------------------------------
    # Raça/cor e escolaridade
    # --------------------------------------------------------

    p3, p4 = st.columns(2)

    with p3:

        ordem_raca = [
            "Parda",
            "Branca",
            "Preta",
            "Amarela",
            "Indígena",
            "Ignorado",
            "Não informado",
        ]

        raca_plot = (
            dados["RACA_COR_DESC"]
            .value_counts()
            .reindex(ordem_raca, fill_value=0)
            .rename_axis("Raça/cor")
            .reset_index(name="Notificações")
        )

        raca_plot["Percentual"] = np.where(
            raca_plot["Notificações"].sum() > 0,
            raca_plot["Notificações"]
            / raca_plot["Notificações"].sum()
            * 100,
            0,
        )

        fig_raca = px.bar(
            raca_plot,
            x="Notificações",
            y="Raça/cor",
            orientation="h",
            color="Raça/cor",
            color_discrete_map=CORES_RACA,
            text="Notificações",
            title="Notificações segundo raça/cor",
            hover_data={"Percentual": ":.1f"},
        )

        fig_raca.update_layout(
            xaxis_title="Número de notificações",
            yaxis_title="",
            showlegend=False,
        )

        estilo_figura(fig_raca, 440)
        st.plotly_chart(fig_raca, use_container_width=True)

        st.caption(
            "Raça/cor é apresentada por frequência e proporção. "
            "A base populacional disponível não possui raça/cor."
        )

    with p4:

        esc_plot = (
            dados["ESCOLARIDADE_DESC"]
            .value_counts()
            .rename_axis("Escolaridade")
            .reset_index(name="Notificações")
            .sort_values("Notificações")
        )

        fig_esc = px.bar(
            esc_plot,
            x="Notificações",
            y="Escolaridade",
            orientation="h",
            color="Notificações",
            color_continuous_scale=[
                "#DCE9F5",
                "#3E6FA8",
                "#173B6C",
            ],
            text="Notificações",
            title="Notificações segundo escolaridade",
        )

        fig_esc.update_layout(
            xaxis_title="Número de notificações",
            yaxis_title="",
            coloraxis_showscale=False,
        )

        estilo_figura(fig_esc, 440)
        st.plotly_chart(fig_esc, use_container_width=True)

        if COL_ESCOLARIDADE is None:
            st.warning(
                "A variável de escolaridade não foi localizada com os "
                "nomes previstos no banco original."
            )

# ============================================================
# 18. CARACTERÍSTICAS DA AUTOLESÃO
# ============================================================

with abas[4]:

    st.subheader("Características das notificações")

    c1, c2 = st.columns([0.8, 1.2])

    # --------------------------------------------------------
    # Recorrência
    # --------------------------------------------------------

    with c1:

        rec_plot = (
            dados["RECORRENCIA_DESC"]
            .value_counts()
            .reindex(
                ["Sim", "Não", "Ignorado", "Não informado"],
                fill_value=0,
            )
            .rename_axis("Recorrência")
            .reset_index(name="Notificações")
        )

        fig_rec = px.bar(
            rec_plot,
            x="Recorrência",
            y="Notificações",
            color="Recorrência",
            color_discrete_map={
                "Sim": REC_SIM,
                "Não": REC_NAO,
                "Ignorado": CINZA,
                "Não informado": "#BFC5CC",
            },
            text="Notificações",
            title="Ocorrência anterior / recorrência",
        )

        fig_rec.update_layout(
            xaxis_title="",
            yaxis_title="Número de notificações",
            showlegend=False,
        )

        estilo_figura(fig_rec, 430)
        st.plotly_chart(fig_rec, use_container_width=True)

        rec_validos = dados[
            dados["RECORRENCIA_DESC"].isin(["Sim", "Não"])
        ]

        if len(rec_validos):

            pct = (
                (rec_validos["RECORRENCIA_DESC"] == "Sim").mean()
                * 100
            )

            st.caption(
                f"Entre {numero_br(len(rec_validos))} registros com "
                f"resposta válida, {percentual_br(pct, 1)} apresentaram "
                f"recorrência."
            )

    # --------------------------------------------------------
    # Métodos
    # --------------------------------------------------------

    with c2:

        if not metodos_long.empty:

            dist_metodo = (
                metodos_long["METODO"]
                .value_counts()
                .rename_axis("Método")
                .reset_index(name="Marcações")
                .sort_values("Marcações")
            )

            fig_metodo = px.bar(
                dist_metodo,
                x="Marcações",
                y="Método",
                orientation="h",
                color="Marcações",
                color_continuous_scale=ESCALA_MAGNITUDE,
                text="Marcações",
                title="Métodos utilizados",
            )

            fig_metodo.update_layout(
                xaxis_title="Número de marcações",
                yaxis_title="",
                coloraxis_showscale=False,
            )

            estilo_figura(fig_metodo, 430)
            st.plotly_chart(fig_metodo, use_container_width=True)

        else:

            st.info("Nenhum método foi identificado na seleção atual.")

    st.caption(
        "Método é uma variável de resposta múltipla. Uma mesma "
        "notificação pode registrar mais de um método; por isso, "
        "a soma das marcações pode superar o total de notificações."
    )

    # --------------------------------------------------------
    # Número de métodos por notificação
    # --------------------------------------------------------

    st.markdown("### Multiplicidade de métodos registrados")

    mult = (
        dados["MULTIPLOS_METODOS"]
        .value_counts()
        .reindex(
            [
                "Nenhum método marcado",
                "Um método",
                "Dois ou mais métodos",
            ],
            fill_value=0,
        )
        .rename_axis("Número de métodos")
        .reset_index(name="Notificações")
    )

    fig_mult = px.bar(
        mult,
        x="Número de métodos",
        y="Notificações",
        color="Número de métodos",
        color_discrete_map={
            "Nenhum método marcado": CINZA,
            "Um método": AZUL2,
            "Dois ou mais métodos": VERMELHO,
        },
        text="Notificações",
        title="Número de métodos registrados por notificação",
    )

    fig_mult.update_layout(
        xaxis_title="",
        yaxis_title="Número de notificações",
        showlegend=False,
    )

    estilo_figura(fig_mult, 390)
    st.plotly_chart(fig_mult, use_container_width=True)

# ============================================================
# 19. ANÁLISE ESTATÍSTICA
# ============================================================

with abas[5]:

    st.subheader("Análise estatística")

    st.caption(
        "Esta área complementa a descrição epidemiológica com "
        "medidas de associação, comparação de taxas e análise "
        "estatística da tendência temporal."
    )

    estat1, estat2, estat3 = st.tabs([
        "Associação",
        "Comparação de taxas",
        "Tendência temporal",
    ])

    # ========================================================
    # 19.1 ASSOCIAÇÃO
    # ========================================================

    with estat1:

        st.markdown("#### Associação entre características das notificações")

        opcoes_associacao = {
            "Sexo": "SEXO_DESC",
            "Faixa etária": "FAIXA_ETARIA",
            "Raça/cor": "RACA_COR_DESC",
            "Escolaridade": "ESCOLARIDADE_DESC",
            "Recorrência": "RECORRENCIA_DESC",
        }

        a1, a2 = st.columns(2)

        with a1:
            var1_nome = st.selectbox(
                "Primeira variável",
                list(opcoes_associacao.keys()),
                index=0,
                key="assoc_var1",
            )

        with a2:
            opcoes_var2 = [
                x for x in opcoes_associacao.keys()
                if x != var1_nome
            ]

            var2_nome = st.selectbox(
                "Segunda variável",
                opcoes_var2,
                index=(
                    opcoes_var2.index("Recorrência")
                    if "Recorrência" in opcoes_var2
                    else 0
                ),
                key="assoc_var2",
            )

        col1 = opcoes_associacao[var1_nome]
        col2 = opcoes_associacao[var2_nome]

        base_assoc = dados[[col1, col2]].copy()

        invalidos = ["Ignorado", "Não informado", "Não se aplica"]

        base_assoc = base_assoc[
            ~base_assoc[col1].isin(invalidos)
            & ~base_assoc[col2].isin(invalidos)
        ].dropna()

        tabela = pd.crosstab(
            base_assoc[col1],
            base_assoc[col2],
        )

        if tabela.shape[0] >= 2 and tabela.shape[1] >= 2:

            chi2, pvalor, v = cramer_v(tabela)

            r1, r2, r3, r4 = st.columns(4)

            r1.metric("N válido", numero_br(len(base_assoc)))
            r2.metric(
                "Qui-quadrado (χ²)",
                numero_br(chi2, 2) if pd.notna(chi2) else "—",
            )
            r3.metric(
                "p-valor",
                (
                    "< 0,001"
                    if pd.notna(pvalor) and pvalor < 0.001
                    else numero_br(pvalor, 3)
                    if pd.notna(pvalor)
                    else "—"
                ),
            )
            r4.metric(
                "V de Cramér",
                numero_br(v, 3) if pd.notna(v) else "—",
            )

            graf_assoc = (
                base_assoc.groupby([col1, col2])
                .size()
                .reset_index(name="N")
            )

            graf_assoc["Percentual"] = (
                graf_assoc.groupby(col1)["N"]
                .transform(lambda x: x / x.sum() * 100)
            )

            fig_assoc = px.bar(
                graf_assoc,
                x=col1,
                y="Percentual",
                color=col2,
                barmode="stack",
                title=f"{var2_nome} segundo {var1_nome.lower()}",
                hover_data={"N": True, "Percentual": ":.1f"},
            )

            fig_assoc.update_layout(
                xaxis_title=var1_nome,
                yaxis_title="Percentual dentro da categoria (%)",
                legend_title=var2_nome,
            )

            estilo_figura(fig_assoc, 440)
            st.plotly_chart(fig_assoc, use_container_width=True)

            st.caption(
                "O teste qui-quadrado avalia evidência de associação "
                "entre as variáveis categóricas. O V de Cramér expressa "
                "a magnitude da associação. Registros ignorados, não "
                "informados e 'não se aplica' são excluídos desta análise."
            )

        else:

            st.warning(
                "A seleção atual não possui categorias suficientes "
                "para realizar o teste de associação."
            )

    # ========================================================
    # 19.2 COMPARAÇÃO DE TAXAS
    # ========================================================

    with estat2:

        st.markdown("#### Comparação de taxas populacionais")

        comparacao = st.radio(
            "Comparar",
            ["Sexo", "Faixa etária"],
            horizontal=True,
            key="comparacao_taxa",
        )

        base_eventos = df.copy()
        base_pop = pop.copy()

        if anos:
            base_eventos = base_eventos[
                base_eventos["NU_ANO"].isin(anos)
            ]
            base_pop = base_pop[
                base_pop["ANO"].isin(anos)
            ]

        if municipios:

            codigos = df.loc[
                df["MUNICÍPIO"].isin(municipios),
                "COD_MUN_6",
            ].unique()

            base_eventos = base_eventos[
                base_eventos["COD_MUN_6"].isin(codigos)
            ]

            base_pop = base_pop[
                base_pop["COD_MUN_6"].isin(codigos)
            ]

        elif regioes:

            codigos = df.loc[
                df["REGIÃO DE SAÚDE"].isin(regioes),
                "COD_MUN_6",
            ].unique()

            base_eventos = base_eventos[
                base_eventos["COD_MUN_6"].isin(codigos)
            ]

            base_pop = base_pop[
                base_pop["COD_MUN_6"].isin(codigos)
            ]

        resultados_taxa = []

        if comparacao == "Sexo":

            definicoes = [
                ("Feminino", 2),
                ("Masculino", 1),
            ]

            for nome, codigo in definicoes:

                n = (
                    base_eventos["SEXO_DESC"]
                    .eq(nome)
                    .sum()
                )

                p = base_pop.loc[
                    base_pop["SEXO"] == codigo,
                    "POPULAÇÃO",
                ].sum()

                taxa = n / p * 100000 if p > 0 else np.nan
                li, ls = ic95_taxa(n, p)

                resultados_taxa.append({
                    "Grupo": nome,
                    "N": n,
                    "População": p,
                    "Taxa": taxa,
                    "LI95": li,
                    "LS95": ls,
                })

        else:

            definicoes = [
                ("10 a 14 anos", list(range(10, 15))),
                ("15 a 19 anos", list(range(15, 20))),
            ]

            for nome, idades_grupo in definicoes:

                n = (
                    base_eventos["IDADE_ANOS"]
                    .isin(idades_grupo)
                    .sum()
                )

                p = base_pop.loc[
                    base_pop["IDADE"].isin(idades_grupo),
                    "POPULAÇÃO",
                ].sum()

                taxa = n / p * 100000 if p > 0 else np.nan
                li, ls = ic95_taxa(n, p)

                resultados_taxa.append({
                    "Grupo": nome,
                    "N": n,
                    "População": p,
                    "Taxa": taxa,
                    "LI95": li,
                    "LS95": ls,
                })

        taxas_df = pd.DataFrame(resultados_taxa)

        g1 = taxas_df.iloc[0]
        g0 = taxas_df.iloc[1]

        rt, rt_li, rt_ls = razao_taxas(
            g1["N"],
            g1["População"],
            g0["N"],
            g0["População"],
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            f"Taxa — {g1['Grupo']}",
            f"{numero_br(g1['Taxa'], 1)}/100 mil",
        )

        m2.metric(
            f"Taxa — {g0['Grupo']}",
            f"{numero_br(g0['Taxa'], 1)}/100 mil",
        )

        m3.metric(
            f"Razão de taxas ({g1['Grupo']} / {g0['Grupo']})",
            numero_br(rt, 2),
        )

        fig_taxas = go.Figure()

        fig_taxas.add_trace(
            go.Scatter(
                x=taxas_df["Taxa"],
                y=taxas_df["Grupo"],
                mode="markers",
                marker=dict(
                    size=12,
                    color=[VERMELHO, AZUL],
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=taxas_df["LS95"] - taxas_df["Taxa"],
                    arrayminus=taxas_df["Taxa"] - taxas_df["LI95"],
                ),
                customdata=np.stack(
                    [
                        taxas_df["N"],
                        taxas_df["LI95"],
                        taxas_df["LS95"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Notificações: %{customdata[0]:,.0f}<br>"
                    "Taxa: %{x:.1f}/100 mil<br>"
                    "IC95%: %{customdata[1]:.1f}–%{customdata[2]:.1f}"
                    "<extra></extra>"
                ),
            )
        )

        fig_taxas.update_layout(
            title="Taxas específicas e intervalos de confiança de 95%",
            xaxis_title="Taxa por 100 mil adolescentes",
            yaxis_title="",
            showlegend=False,
        )

        estilo_figura(fig_taxas, 350)
        st.plotly_chart(fig_taxas, use_container_width=True)

        if pd.notna(rt):

            st.markdown(
                f"**Razão de taxas:** {numero_br(rt, 2)} "
                f"(IC95% {numero_br(rt_li, 2)}–{numero_br(rt_ls, 2)})."
            )

        st.caption(
            "As comparações utilizam apenas dimensões para as quais "
            "a base populacional possui denominadores compatíveis."
        )

    # ========================================================
    # 19.3 TENDÊNCIA TEMPORAL
    # ========================================================

    with estat3:

        st.markdown("#### Tendência estatística das taxas anuais")

        st.caption(
            "A tendência é estimada sobre as taxas anuais de notificação. "
            "O modelo abaixo utiliza regressão log-linear das taxas como "
            "síntese descritiva da variação temporal."
        )

        serie_eventos = (
            df.groupby("NU_ANO")
            .size()
            .reindex(range(2014, 2025), fill_value=0)
            .rename("Notificações")
            .reset_index()
            .rename(columns={"index": "Ano"})
        )

        if "NU_ANO" in serie_eventos.columns:
            serie_eventos = serie_eventos.rename(
                columns={"NU_ANO": "Ano"}
            )

        pop_anual = (
            pop.groupby("ANO", as_index=False)["POPULAÇÃO"]
            .sum()
            .rename(columns={"ANO": "Ano"})
        )

        tendencia = serie_eventos.merge(
            pop_anual,
            on="Ano",
            how="left",
        )

        tendencia["Taxa"] = (
            tendencia["Notificações"]
            / tendencia["POPULAÇÃO"]
            * 100000
        )

        tendencia_modelo = tendencia[
            tendencia["Taxa"] > 0
        ].copy()

        try:

            from scipy.stats import linregress

            x = tendencia_modelo["Ano"].astype(float)
            y = np.log(tendencia_modelo["Taxa"].astype(float))

            resultado = linregress(x, y)

            beta = resultado.slope
            se_beta = resultado.stderr

            vap = (np.exp(beta) - 1) * 100

            li_vap = (
                np.exp(beta - 1.96 * se_beta) - 1
            ) * 100

            ls_vap = (
                np.exp(beta + 1.96 * se_beta) - 1
            ) * 100

            p_tendencia = resultado.pvalue

            t1, t2, t3 = st.columns(3)

            t1.metric(
                "Variação percentual anual",
                percentual_br(vap, 1),
            )

            t2.metric(
                "IC95%",
                (
                    f"{numero_br(li_vap, 1)}% a "
                    f"{numero_br(ls_vap, 1)}%"
                ),
            )

            t3.metric(
                "p-valor",
                (
                    "< 0,001"
                    if p_tendencia < 0.001
                    else numero_br(p_tendencia, 3)
                ),
            )

            tendencia["Taxa estimada"] = np.exp(
                resultado.intercept
                + resultado.slope * tendencia["Ano"]
            )

            fig_tend = go.Figure()

            fig_tend.add_trace(
                go.Scatter(
                    x=tendencia["Ano"],
                    y=tendencia["Taxa"],
                    mode="lines+markers",
                    name="Taxa observada",
                    line=dict(
                        color=VERMELHO,
                        width=3,
                    ),
                )
            )

            fig_tend.add_trace(
                go.Scatter(
                    x=tendencia["Ano"],
                    y=tendencia["Taxa estimada"],
                    mode="lines",
                    name="Tendência estimada",
                    line=dict(
                        color=AZUL,
                        width=2,
                        dash="dash",
                    ),
                )
            )

            fig_tend.update_layout(
                title="Taxa observada e tendência temporal estimada",
                xaxis_title="Ano",
                yaxis_title="Taxa por 100 mil adolescentes",
                hovermode="x unified",
            )

            estilo_figura(fig_tend, 430)
            st.plotly_chart(fig_tend, use_container_width=True)

            st.caption(
                "A variação percentual anual resume a direção e a "
                "magnitude média da tendência no período. O resultado "
                "deve ser interpretado como tendência das notificações "
                "registradas, e não como incidência clínica de autolesão."
            )

        except Exception as erro:

            st.warning(
                "Não foi possível calcular a tendência estatística."
            )
            st.caption(str(erro))

# ============================================================
# 20. QUALIDADE DOS DADOS
# ============================================================

with abas[6]:

    st.subheader("Qualidade e completude da informação")

    def completude(base, coluna, validos=None):

        if validos is None:

            invalidos = [
                "Ignorado",
                "Não informado",
                "Não se aplica",
            ]

            return (
                ~base[coluna].isin(invalidos)
                & base[coluna].notna()
            ).mean() * 100

        return base[coluna].isin(validos).mean() * 100

    indicadores_qualidade = pd.DataFrame({
        "Variável": [
            "Sexo",
            "Raça/cor",
            "Escolaridade",
            "Recorrência",
            "Município",
            "Região de Saúde",
        ],
        "Completude (%)": [
            completude(
                dados,
                "SEXO_DESC",
                ["Feminino", "Masculino"],
            ),
            completude(dados, "RACA_COR_DESC"),
            completude(dados, "ESCOLARIDADE_DESC"),
            completude(
                dados,
                "RECORRENCIA_DESC",
                ["Sim", "Não"],
            ),
            dados["MUNICÍPIO"].notna().mean() * 100,
            dados["REGIÃO DE SAÚDE"].notna().mean() * 100,
        ],
    })

    indicadores_qualidade = indicadores_qualidade.sort_values(
        "Completude (%)"
    )

    fig_comp = px.bar(
        indicadores_qualidade,
        x="Completude (%)",
        y="Variável",
        orientation="h",
        color="Completude (%)",
        color_continuous_scale=[
            "#E15759",
            "#F28E2B",
            "#FDCB6E",
            "#59A14F",
        ],
        text="Completude (%)",
        title="Completude das principais variáveis",
        range_x=[0, 100],
    )

    fig_comp.update_traces(
        texttemplate="%{text:.1f}%",
    )

    fig_comp.update_layout(
        xaxis_title="Completude (%)",
        yaxis_title="",
        coloraxis_showscale=False,
    )

    estilo_figura(fig_comp, 430)
    st.plotly_chart(fig_comp, use_container_width=True)

    # --------------------------------------------------------
    # Completude por ano
    # --------------------------------------------------------

    linhas = []

    for ano, base_ano in dados.groupby("NU_ANO"):

        linhas.extend([
            {
                "Ano": ano,
                "Variável": "Sexo",
                "Completude": completude(
                    base_ano,
                    "SEXO_DESC",
                    ["Feminino", "Masculino"],
                ),
            },
            {
                "Ano": ano,
                "Variável": "Raça/cor",
                "Completude": completude(
                    base_ano,
                    "RACA_COR_DESC",
                ),
            },
            {
                "Ano": ano,
                "Variável": "Escolaridade",
                "Completude": completude(
                    base_ano,
                    "ESCOLARIDADE_DESC",
                ),
            },
            {
                "Ano": ano,
                "Variável": "Recorrência",
                "Completude": completude(
                    base_ano,
                    "RECORRENCIA_DESC",
                    ["Sim", "Não"],
                ),
            },
        ])

    comp_ano = pd.DataFrame(linhas)

    if not comp_ano.empty:

        matriz_comp = comp_ano.pivot(
            index="Variável",
            columns="Ano",
            values="Completude",
        )

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=matriz_comp.values,
                x=matriz_comp.columns,
                y=matriz_comp.index,
                colorscale=[
                    [0.0, "#E15759"],
                    [0.5, "#FDCB6E"],
                    [1.0, "#59A14F"],
                ],
                zmin=0,
                zmax=100,
                text=np.round(matriz_comp.values, 1),
                texttemplate="%{text}%",
                hovertemplate=(
                    "Ano: %{x}<br>"
                    "Variável: %{y}<br>"
                    "Completude: %{z:.1f}%"
                    "<extra></extra>"
                ),
                colorbar=dict(title="Completude (%)"),
            )
        )

        fig_heat.update_layout(
            title="Completude segundo variável e ano de notificação",
            xaxis_title="Ano",
            yaxis_title="",
        )

        estilo_figura(fig_heat, 400)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.caption(
        "A análise de completude permite avaliar a qualidade do "
        "preenchimento das variáveis ao longo do período. Categorias "
        "ignoradas, não informadas e, quando pertinente, 'não se aplica' "
        "não são consideradas informação completa."
    )

# ============================================================
# 21. NOTAS METODOLÓGICAS
# ============================================================

st.divider()

with st.expander("Notas metodológicas"):

    st.markdown(
        """
**População do estudo**

Notificações de lesão autoprovocada em adolescentes de **10 a 19 anos**,
residentes em Pernambuco, registradas entre **2014 e 2024**.

**Fonte das notificações**

Sistema de Informação de Agravos de Notificação (SINAN).

**Universo validado**

O processamento deve reproduzir **12.713 notificações**. O aplicativo
interrompe o carregamento se esse total não for reproduzido antes da
aplicação dos filtros.

**Taxas**

As taxas são calculadas por 100 mil adolescentes utilizando a população
correspondente ao ano, município, sexo e idade quando essas dimensões
estão disponíveis na base populacional.

Quando são aplicados filtros para **raça/cor, escolaridade, recorrência
ou método**, a taxa populacional da seleção não é calculada, porque não
existem denominadores correspondentes na base populacional utilizada.

Para períodos com mais de um ano, o cálculo utiliza a soma das
notificações dividida pela soma das populações anuais, representando uma
taxa média sobre pessoas-tempo no período selecionado.

**Intervalos de confiança**

Os IC95% apresentados para taxas utilizam aproximação baseada na
distribuição de Poisson. Em municípios com pequeno número de eventos,
as estimativas podem apresentar maior instabilidade.

**Raça/cor e escolaridade**

São analisadas por números, proporções e completude, e não por taxas
populacionais, devido à ausência de denominadores específicos.

**Recorrência**

A proporção de recorrência utiliza como denominador apenas registros
com respostas válidas **Sim/Não**. Ignorados e não informados são
preservados para avaliação da qualidade da informação.

**Métodos utilizados**

Os campos de método constituem **resposta múltipla**. Uma mesma
notificação pode registrar mais de um método. Portanto, a soma das
marcações pode ser superior ao número total de notificações.

**Análises estatísticas**

O teste qui-quadrado é utilizado para investigar associação entre
variáveis categóricas, acompanhado pelo **V de Cramér** como medida de
magnitude da associação. Categorias ignoradas, não informadas e
não aplicáveis são retiradas dessas comparações.

As razões de taxas comparam grupos para os quais existem denominadores
populacionais compatíveis. A análise de tendência temporal utiliza a
transformação logarítmica das taxas anuais para estimar uma variação
percentual anual descritiva e seu intervalo de confiança.

Os resultados descrevem **notificações registradas no sistema de
vigilância**. Não devem ser interpretados automaticamente como
incidência real de todos os episódios de autolesão na população.
        """
    )

# ============================================================
# 22. PRÉ-VISUALIZAÇÃO DOS DADOS
# ============================================================

with st.expander("Pré-visualização dos dados"):

    st.caption(
        "São exibidas somente variáveis analíticas do painel. "
        "Identificadores desnecessários não são apresentados."
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
        "MULTIPLOS_METODOS",
    ]

    preview = dados[
        [c for c in colunas_preview if c in dados.columns]
    ].copy()

    preview = preview.rename(columns={
        "NU_ANO": "Ano",
        "IDADE_ANOS": "Idade",
        "FAIXA_ETARIA": "Faixa etária",
        "SEXO_DESC": "Sexo",
        "RACA_COR_DESC": "Raça/cor",
        "ESCOLARIDADE_DESC": "Escolaridade",
        "RECORRENCIA_DESC": "Recorrência",
        "MUNICÍPIO": "Município",
        "REGIÃO DE SAÚDE": "Região de Saúde",
        "N_METODOS": "Nº de métodos",
        "MULTIPLOS_METODOS": "Classificação dos métodos",
    })

    st.dataframe(
        preview.head(500),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Mostrando até 500 registros de "
        f"{numero_br(len(preview))} registros na seleção atual."
    )

# ============================================================
# 23. RODAPÉ
# ============================================================

st.divider()

st.caption(
    "Fonte: Sistema de Informação de Agravos de Notificação (SINAN) "
    "e base populacional utilizada no estudo. Elaboração própria."
)

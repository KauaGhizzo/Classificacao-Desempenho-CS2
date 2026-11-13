"""Classifica o desempenho individual de jogadores de CS2 em mapas concluídos."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    precision_recall_fscore_support, silhouette_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

RAIZ = Path(__file__).resolve().parent
ARQUIVO_DADOS = RAIZ / "dados" / "cs2_all_tiers_games.csv"
SEMENTE = 42
LIMIAR = 0.15
CLASSES = ["Baixo", "Intermediário", "Alto"]
# Usados pelos modelos. Eliminações, mortes e saldo servem apenas ao rótulo.
ATRIBUTOS = ["adr", "kast", "assist_por_round"]


def ler_jogadores(caminho):
    """Converte os 10 blocos de colunas em observações de jogador por mapa."""
    dados = pd.read_csv(caminho)
    placar = dados[["score1_game", "score2_game"]].apply(pd.to_numeric, errors="coerce")
    datas = pd.to_datetime(dados["datetime"], utc=True, errors="coerce")
    mapas = dados["is_total"].astype(str).str.lower().eq("false")
    ids = pd.to_numeric(dados["game_id"], errors="coerce")
    validos = (
        mapas & ids.gt(0) & dados["match_id"].notna() & dados["map_name"].notna()
        & datas.notna() & placar.ge(0).all(axis=1) & placar.max(axis=1).ge(13)
        & placar["score1_game"].ne(placar["score2_game"])
    )
    base = dados.loc[validos].copy()
    base["dia"] = datas.loc[validos].dt.normalize()
    base["rounds"] = placar.loc[validos].sum(axis=1)
    base = base.drop_duplicates(["match_id", "game_id"])
    chaves = ["match_id", "game_id", "map_name", "dia", "rounds"]
    campos = ["id", "kills", "deaths", "assists", "adr", "kast"]
    blocos = []
    for equipe in (1, 2):
        for posicao in range(1, 6):
            prefixo = f"team{equipe}_player{posicao}"
            renomear = {f"{prefixo}_{campo}": campo for campo in campos}
            renomear[prefixo] = "jogador"
            renomear[f"team{equipe}"] = "equipe"
            blocos.append(base[chaves + list(renomear)].rename(columns=renomear))

    jogadores = pd.concat(blocos, ignore_index=True)
    jogadores[campos] = jogadores[campos].apply(pd.to_numeric, errors="coerce")
    completos = np.isfinite(jogadores[campos]).all(axis=1)
    validos = (
        completos & jogadores["id"].gt(0) & jogadores["kast"].between(0, 100)
        & jogadores[["kills", "deaths", "assists", "adr"]].ge(0).all(axis=1)
    )
    jogadores = jogadores.loc[validos].drop_duplicates(["match_id", "game_id", "id"]).copy()
    jogadores["id"] = jogadores["id"].astype("int64")
    jogadores["assist_por_round"] = jogadores["assists"] / jogadores["rounds"]
    saldo = (jogadores["kills"] - jogadores["deaths"]) / jogadores["rounds"]
    jogadores["saldo_por_round"] = saldo
    jogadores["classe"] = np.select([saldo.lt(-LIMIAR), saldo.gt(LIMIAR)], [0, 2], default=1)
    resumo = {
        "linhas_csv": len(dados),
        "colunas_csv": len(dados.columns),
        "linhas_por_mapa": int(mapas.sum()),
        "mapas_apos_filtros": len(base),
        "jogador_mapa_validos": len(jogadores),
        "jogadores_apos_filtros": int(jogadores["id"].nunique()),
    }
    return jogadores, resumo


def dividir_por_tempo(dados):
    dias = dados["dia"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(dias) < 10:
        raise ValueError("A avaliação exige pelo menos 10 dias com mapas válidos.")
    corte = dias.iloc[int(len(dias) * 0.8)]
    treino = dados.loc[dados["dia"] < corte].reset_index(drop=True)
    teste = dados.loc[dados["dia"] >= corte].reset_index(drop=True)
    dias_treino = dias.loc[dias < corte]
    folds = []
    for passado, futuro in TimeSeriesSplit(n_splits=3).split(dias_treino):
        antes = np.flatnonzero(treino["dia"].isin(dias_treino.iloc[passado]))
        depois = np.flatnonzero(treino["dia"].isin(dias_treino.iloc[futuro]))
        if treino.iloc[antes]["classe"].nunique() != len(CLASSES):
            raise ValueError("Um período de treino não contém todas as classes.")
        folds.append((antes, depois))
    return treino, teste, folds


def salvar_figura(fig, saida, nome):
    fig.savefig(saida / nome, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def avaliar_classificadores(treino, teste, folds, saida, jobs):
    x_treino, y_treino = treino[ATRIBUTOS], treino["classe"]
    x_teste, y_teste = teste[ATRIBUTOS], teste["classe"]
    configuracoes = {
        "SVM linear": (
            make_pipeline(StandardScaler(), LinearSVC(random_state=SEMENTE, max_iter=5000)),
            {"linearsvc__C": [0.01, 0.1, 1.0], "linearsvc__class_weight": [None, "balanced"]},
        ),
        "Árvore de decisão": (
            DecisionTreeClassifier(random_state=SEMENTE, class_weight="balanced"),
            {"max_depth": [3, 6, 10], "min_samples_leaf": [50, 200]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=80, random_state=SEMENTE, class_weight="balanced"),
            {"max_depth": [5, 10], "min_samples_leaf": [50, 150]},
        ),
    }
    modelos = {"Classe majoritária": DummyClassifier(strategy="most_frequent")}
    modelos.update({
        nome: GridSearchCV(modelo, grade, scoring="f1_macro", cv=folds, n_jobs=jobs, error_score="raise")
        for nome, (modelo, grade) in configuracoes.items()
    })
    identificadores = ["dia", "match_id", "game_id", "map_name", "id", "jogador", "equipe"]
    classificacoes = teste[identificadores + ATRIBUTOS + ["saldo_por_round"]].copy()
    classificacoes["dia"] = classificacoes["dia"].dt.date
    classificacoes["classe_real"] = y_teste.map(dict(enumerate(CLASSES)))
    relatorios = {}
    resultados = []
    rotulos = list(range(len(CLASSES)))
    fig, eixos = plt.subplots(2, 2, figsize=(9.5, 7.5), layout="constrained")
    for (nome, modelo), eixo in zip(modelos.items(), eixos.flat):
        print(f"Treinando {nome}...", flush=True)
        modelo.fit(x_treino, y_treino)
        classificado = modelo.predict(x_teste)
        classificacoes[nome] = np.asarray(CLASSES)[classificado]
        precisao, recall, f1, _ = precision_recall_fscore_support(
            y_teste, classificado, labels=rotulos, average="macro", zero_division=0,
        )
        resultados.append({
            "modelo": nome, "acuracia": accuracy_score(y_teste, classificado),
            "precisao_macro": precisao, "recall_macro": recall, "f1_macro": f1,
            "f1_macro_validacao": getattr(modelo, "best_score_", None),
        })
        relatorios[nome] = {
            "parametros": getattr(modelo, "best_params_", {}),
            "por_classe": classification_report(
                y_teste, classificado, labels=rotulos, target_names=CLASSES,
                zero_division=0, output_dict=True,
            ),
        }
        exibicao = ConfusionMatrixDisplay.from_predictions(
            y_teste, classificado, labels=rotulos, display_labels=CLASSES, normalize="true",
            cmap="Blues", colorbar=False, values_format=".0%", ax=eixo,
            im_kw={"vmin": 0, "vmax": 1},
        )
        for valor, texto in zip(exibicao.confusion_matrix.flat, exibicao.text_.flat):
            texto.set_color("white" if valor > 0.55 else "#142339")
        eixo.set(title=nome, xlabel="Classe atribuída pelo modelo", ylabel="Classe real")
    fig.suptitle("Matrizes de confusão no teste | porcentagem por classe real", fontsize=14)
    salvar_figura(fig, saida, "matrizes_confusao.png")
    resultados = pd.DataFrame(resultados).sort_values("f1_macro", ascending=False)
    resultados.to_csv(saida / "metricas_classificacao.csv", index=False)
    classificacoes.to_csv(saida / "classificacoes_teste.csv", index=False, encoding="utf-8-sig")
    (saida / "metricas_por_classe.json").write_text(
        json.dumps(relatorios, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return resultados


def explorar_grupos(treino, saida):
    """Agrupa observações por mapa com base nas estatísticas individuais do mapa."""
    perfis = treino[ATRIBUTOS]
    escala = StandardScaler()
    x = escala.fit_transform(perfis)
    amostra = perfis.sample(n=min(2500, len(perfis)), random_state=SEMENTE).index.to_numpy()
    candidatos = {}
    pontuacoes = []
    for k in range(2, 7):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit(x)
        silhueta = silhouette_score(x[amostra], modelo.labels_[amostra])
        candidatos[k] = modelo
        pontuacoes.append({"k": k, "silhueta": silhueta, "inercia": modelo.inertia_})
    pontuacoes = pd.DataFrame(pontuacoes)
    melhor = pontuacoes.loc[pontuacoes["silhueta"].idxmax()]
    escolhido = candidatos[int(melhor["k"])]
    pontuacoes.to_csv(saida / "avaliacao_kmeans.csv", index=False)

    grupos = pd.Series(escolhido.labels_ + 1, name="grupo", index=perfis.index)
    resumo = perfis.groupby(grupos).mean()
    resumo.insert(0, "observacoes", grupos.value_counts())
    # Os rótulos entram somente aqui, para descrever os grupos já formados.
    proporcoes = pd.crosstab(grupos, treino["classe"], normalize="index")
    proporcoes = proporcoes.reindex(columns=range(len(CLASSES)), fill_value=0)
    proporcoes.columns = ["proporcao_baixo", "proporcao_intermediario", "proporcao_alto"]
    resumo.join(proporcoes).to_csv(saida / "perfis_kmeans.csv")

    pca = PCA(n_components=2).fit(x)
    pontos = pca.transform(x[amostra])
    centroides = pca.transform(escolhido.cluster_centers_)
    variancia = pca.explained_variance_ratio_
    fig, (esq, dir_) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    esq.plot(pontuacoes["k"], pontuacoes["silhueta"], "o-", color="#1d689a")
    esq.set(title="Escolha do número de grupos", xlabel="Número de grupos (k)", ylabel="Silhueta média")
    esq.set_xticks(pontuacoes["k"])
    cores = ["#177589", "#c85e31", "#5c69b1", "#449356", "#b6528a", "#8b6d3c"]
    for grupo in range(escolhido.n_clusters):
        pertence = escolhido.labels_[amostra] == grupo
        dir_.scatter(*pontos[pertence].T, s=7, alpha=0.35, color=cores[grupo], label=f"Grupo {grupo + 1}")
        dir_.scatter(*centroides[grupo], s=55, marker="X", color=cores[grupo], edgecolor="#132238")
    dir_.set(
        title="Observações projetadas por PCA",
        xlabel=f"Componente 1 ({variancia[0]:.0%})",
        ylabel=f"Componente 2 ({variancia[1]:.0%})",
    )
    dir_.legend(markerscale=2)
    salvar_figura(fig, saida, "kmeans.png")
    return {"k": escolhido.n_clusters, "silhueta": float(melhor["silhueta"]), "observacoes": len(perfis)}


def descrever_base(treino, teste, saida):
    distribuicao = pd.DataFrame({
        "treino": treino["classe"].value_counts(),
        "teste": teste["classe"].value_counts(),
    }).reindex(range(len(CLASSES)), fill_value=0).fillna(0).astype(int).set_axis(CLASSES)
    distribuicao.to_csv(saida / "distribuicao_classes.csv", index_label="classe")
    proporcoes = distribuicao.div(distribuicao.sum()) * 100
    eixo = proporcoes.plot.bar(rot=0, color=["#1d689a", "#d47b3b"], figsize=(7, 4))
    for barras in eixo.containers:
        eixo.bar_label(barras, fmt="%.1f%%", padding=3)
    eixo.set(title="Distribuição da variável-alvo", ylabel="Percentual de observações", ylim=(0, float(proporcoes.max().max()) * 1.2))
    salvar_figura(eixo.figure, saida, "distribuicao_classes.png")
    treino[ATRIBUTOS].describe().to_csv(saida / "descricao_atributos_treino.csv")


def executar(caminho, saida, jobs):
    saida.mkdir(parents=True, exist_ok=True)
    dados, resumo = ler_jogadores(caminho)
    treino, teste, folds = dividir_por_tempo(dados)
    resumo.update({
        "jogador_mapa_modelagem": len(dados),
        "jogadores_modelagem": int(dados["id"].nunique()),
        "jogador_mapa_treino": len(treino),
        "jogador_mapa_teste": len(teste),
        "inicio_treino_utc": treino["dia"].min().date().isoformat(),
        "fim_treino_utc": treino["dia"].max().date().isoformat(),
        "inicio_teste_utc": teste["dia"].min().date().isoformat(),
        "fim_teste_utc": teste["dia"].max().date().isoformat(),
    })
    descrever_base(treino, teste, saida)
    with threadpool_limits(limits=2):
        metricas = avaliar_classificadores(treino, teste, folds, saida, jobs)
        resumo["kmeans"] = explorar_grupos(treino, saida)
    (saida / "resumo_execucao.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("\n", metricas.round(4).to_string(index=False), sep="")
    print(f"\nK-means: {resumo['kmeans']}")
    print(f"Resultados em {saida.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ARQUIVO_DADOS, help="CSV original do Kaggle")
    parser.add_argument("--saida", type=Path, default=RAIZ / "resultados", help="Pasta dos resultados")
    parser.add_argument("--jobs", type=int, default=1, help="Processos paralelos na busca de parâmetros")
    args = parser.parse_args()
    if not args.dados.is_file():
        parser.error(f"CSV não encontrado: {args.dados}")
    executar(args.dados, args.saida, args.jobs)


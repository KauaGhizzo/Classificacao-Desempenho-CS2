# Classificação de desempenho de jogadores profissionais de CS2

Projeto acadêmico de machine learning em Python. Compara três classificadores supervisionados e usa K-means para explorar perfis de desempenho.

## Objetivo

Classificar o desempenho individual de um jogador profissional **depois que o mapa termina**, usando as estatísticas do próprio mapa. Cada linha da base preparada representa um jogador em um mapa. Um mesmo jogador pode receber classes diferentes em mapas diferentes.

| Classe | Saldo de eliminações por round |
| --- | --- |
| Baixo | Menor que −0,15 |
| Intermediário | Entre −0,15 e +0,15, inclusive |
| Alto | Maior que +0,15 |

O saldo é calculado como **(eliminações − mortes) ÷ total de rounds**. Os limites são uma definição adotada no projeto. Em um mapa de 20 rounds, saldo de +3 é intermediário e saldo de +4 é alto. A classe representa o desempenho segundo esse critério.

## Metodologia

1. O arquivo `cs2_all_tiers_games.csv` é filtrado para reter os mapas individuais com placar válido. Os dez blocos de estatísticas são convertidos em observações por jogador e mapa. Duplicatas e linhas sem as estatísticas necessárias são removidas.
2. Os modelos recebem **ADR** (dano médio por round), **KAST** (percentual de rounds com eliminação, assistência, sobrevivência ou troca) e **assistências por round**. A classe de referência é calculada a partir de eliminações e mortes. O saldo e suas duas contagens ficam fora dos atributos dos modelos para evitar vazamento do rótulo. Quando o objetivo é obter exatamente a faixa definida acima e as duas contagens estão disponíveis, a fórmula já determina o resultado. O experimento mede quanto as outras três estatísticas ajudam a reproduzir essa faixa.
3. O teste reúne os 20% mais recentes das datas válidas, com arredondamento em dias. Os hiperparâmetros são escolhidos em três divisões temporais do treino. A SVM é padronizada dentro de cada divisão. Um jogador pode aparecer tanto no treino quanto no teste, em mapas diferentes.
4. Os classificadores são **SVM linear** (`LinearSVC`), **Árvore de Decisão** e **Random Forest** (ensemble de árvores). O **F1 macro** é a métrica principal e dá o mesmo peso às três classes. A classe majoritária do treino serve de referência. A avaliação também inclui acurácia, precisão macro, recall macro e métricas por classe.
5. O **K-means** agrupa as observações por jogador e mapa no treino usando as três entradas padronizadas. São comparados valores de k de 2 a 6 pela silhueta média em uma amostra fixa. A projeção em duas dimensões usa PCA para visualização. A proporção de cada classe dentro de cada grupo é calculada após o ajuste.

## Resultados da execução incluída

O teste contém **23.981 observações de jogador por mapa**.

| Modelo | Acurácia | F1 macro |
| --- | ---: | ---: |
| Random Forest | 71,33% | 71,97% |
| Árvore de decisão | 70,57% | 71,18% |
| SVM linear | 69,98% | 70,51% |
| Classe majoritária | 39,56% | 18,90% |

A **Random Forest** teve o maior F1 macro (71,97%) e a maior acurácia (71,33%). A referência baseada na classe majoritária obteve 39,56% de acurácia. No K-means, 2 grupos atingiram a maior silhueta média (0,321).

## Como executar no Windows

Extraia o ZIP e abra o terminal na pasta do projeto. Use Python 3.12.

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 desempenho_cs2_ml.py
```

Se `py` estiver indisponível e `python --version` apontar para Python 3.12, use `python` nos dois comandos. O CSV já está em `dados/`. Passe `--saida nova_execucao` para salvar novos resultados em outra pasta.

## Como executar no Google Colab

Envie o ZIP para a sessão do Colab e rode a célula abaixo. Se o arquivo enviado tiver outro nome, ajuste o primeiro comando.

```python
!unzip -q -o Classificacao_Desempenho_CS2.zip
%cd Classificacao_Desempenho_CS2
%pip install -q -r requirements.txt
!python desempenho_cs2_ml.py --saida nova_execucao
```

## Arquivos

- `desempenho_cs2_ml.py` reúne preparação, treinamento, avaliação e geração dos gráficos.
- `Relatorio_Desempenho_CS2.pdf` apresenta o experimento e interpreta os resultados.
- `resultados/classificacoes_teste.csv` registra o rótulo real, a classe atribuída por cada modelo, o saldo de referência e as entradas de cada observação do teste.
- `resultados/` também contém métricas, parâmetros escolhidos, matrizes de confusão, distribuição de classes, silhueta e perfis dos grupos.
- `dados/cs2_all_tiers_games.csv` contém o CSV original, sem alterações.

## Fonte dos dados

**ektarr. Counter-Strike Pro Matches.** Versão 10, Kaggle. O arquivo original incluído está licenciado sob [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). [Página do dataset](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches/data).


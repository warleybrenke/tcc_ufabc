# Simulações do TCC: quadrirrotor com controle PID

Este código reproduz os quatro ensaios numéricos descritos no trabalho
*Modelagem pelas equações de Newton–Euler e controle PID de um VANT quadrirrotor*,
de Warley Brenke Diniz da Silva (UFABC, 2026):

1. subida de 0 a 5 m seguida de voo pairado;
2. deslocamento horizontal para x = 2 m e y = 1,5 m;
3. comando de guinada de 15 graus;
4. recuperação do voo pairado após perturbações iniciais.

## Como executar

Requer Python 3, NumPy e Matplotlib. Na pasta que contém
simulacao_quadrirotor.py e requirements.txt:

~~~sh
python -m pip install -r requirements.txt
python simulacao_quadrirotor.py
~~~

Por padrão, o script cria as quatro figuras usadas no texto e o arquivo
figs/metricas_simulacao.csv. Para escolher outra pasta:

~~~sh
python simulacao_quadrirotor.py --output-dir resultados
~~~

Os resultados do texto foram reproduzidos com Python 3.14.2, NumPy 2.4.1 e
Matplotlib 3.10.8. A simulação usa passo de 0,002 s e integração RK4. O script
imprime o posto da matriz de controlabilidade, os ganhos e polos projetados, as
métricas de desempenho e a contagem de amostras com saturação no ensaio de
recuperação.

## Escopo e limites

O modelo é linearizado em torno do voo pairado. A trajetória de subida não
representa contato com o solo, efeito solo nem a dinâmica não linear completa.
Os parâmetros físicos foram adotados de Luukkonen (2011); o limite de
900 rad/s para cada rotor é uma hipótese numérica, não a especificação de um
conjunto motor–hélice identificado experimentalmente. Portanto, os ensaios
avaliam a configuração modelada e não demonstram viabilidade de voo real.

Os critérios e definições das métricas estão no Capítulo 4 do TCC. Em especial,
o sobressinal e o tempo de acomodação são calculados após o término de cada
transição cúbica de referência; não são uma medição de resposta a degrau puro.

## Referência do modelo

Luukkonen, T. *Modelling and control of quadcopter*. Aalto University, 2011.
Consulte a bibliografia do TCC para a referência completa e os demais trabalhos
relacionados.

## Repositório e versão

O endereço deste código é
https://github.com/warleybrenke/tcc_ufabc. O link para o texto final do TCC
será acrescentado após o depósito. A versão utilizada no texto depositado
deverá ser identificada por um commit ou uma tag Git.

Não há licença de reutilização atribuída nesta versão. A publicação permite
consultar o código, mas não concede uma autorização geral para copiá-lo,
modificá-lo ou redistribuí-lo. O autor poderá definir uma licença futuramente.
A publicação do código não substitui a checagem independente dos resultados.

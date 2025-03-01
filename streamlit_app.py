import streamlit as st
import itertools
import pandas as pd
import pulp as pl
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from scipy.optimize import differential_evolution
from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')

columns_config = ['ticker', 'strike', 'valor']
full_columns_config = ['operacao', 'ticker', 'strike', 'valor']

def calculate_entry(call_df, put_df):
    somaproduto_call = int((call_df['valor'] * call_df['quantidade']).sum())
    somaproduto_put  = int((put_df['valor'] * put_df['quantidade']).sum())
    return somaproduto_call + somaproduto_put

def calculate_rt(strike_final, valor_entrada_op, call_strikes, call_qtd, put_strikes, put_qtd):
    result = 0
    for D, E in zip(call_strikes, call_qtd):
        result += max(strike_final - D, 0) * E
    for M, N in zip(put_strikes, put_qtd):
        result += max(M - strike_final, 0) * N
    result -= valor_entrada_op
    return float(result)

def calcula_pesos(strike, center, points):
        return -(abs((strike-center)**2/(points)))

# Função para calcular o coeficiente de cada opção (payoff líquido)
def payoff_coefficient(row, strike_regua):
    op = row['operacao']
    strike = row['strike']
    valor = row['valor']
    if op == 'CALL':
        return max(strike_regua - strike, 0) - valor
    elif op == 'PUT':
        return max(strike - strike_regua, 0) - valor
    else:
        return 0

#####################################################################

st.title('Otimização de derivativos')

col1, col2 = st.columns(2)

with col1:
    st.header("Call")
    template_call_df = pd.DataFrame([[None, None, None]], columns=columns_config)
    edit_call_df = st.data_editor(template_call_df, num_rows="dynamic", key = 1)

with col2:
    st.header("Put")
    template_put_df = pd.DataFrame([[None, None, None]], columns=columns_config)
    edit_put_df = st.data_editor(template_put_df, num_rows="dynamic", key = 2)


option = st.selectbox(
    "Qual código você deseja executar?",
    ("Otimização Personalizada", "Otimização Gaussiana", "Otimização Linear"),
)

#####################################################################

if option == "Otimização Personalizada":

    col1, col2, col3 = st.columns(3)
    
    with col1:
        left_input = st.text_input("Strike Minimo", "113")

    with col2:
        center_input = st.text_input("Strike Central", "123")

    with col3:
        right_input = st.text_input("Strike Maximo", "133")
    
    col1, col2, col3 , col4, col5, col6, col7 = st.columns(7)
    if col4.button("Calcular", type="primary"):

        call_df = edit_call_df.copy()
        put_df = edit_put_df.copy()

        call_df['operacao'] = "CALL"
        put_df['operacao'] = "PUT"

        
        call_df['strike'] = call_df['strike'].apply(lambda x: x.replace(',', '.'))
        call_df['valor'] = call_df['valor'].apply(lambda x: x.replace(',', '.'))
        put_df['strike'] = put_df['strike'].apply(lambda x: x.replace(',', '.'))
        put_df['valor'] = put_df['valor'].apply(lambda x: x.replace(',', '.'))

        call_df['strike'] = call_df['strike'].astype(float)
        call_df['valor'] = call_df['valor'].astype(float)
        put_df['strike'] = put_df['strike'].astype(float)
        put_df['valor'] = put_df['valor'].astype(float)

        market_df = pd.concat([call_df, put_df])

        strike_central = abs(float(center_input.replace(',', '.'))) 
        strike_min = abs(float(left_input.replace(',', '.')))
        strike_max = abs(float(right_input.replace(',', '.')))  
        num_points = 50

        def objective(quantities): 
            alpha=1.0            # Peso para o retorno no strike central, 
            beta=1.0             # Peso para o retorno ponderado, 
            penalty_multiplier=1000      # Multiplicador para penalizar se o retorno central não for máximo
            penalty_entry_multiplier=1000  # Multiplicador para penalizar se a arrecadação for > 0):
            
            # Arredonda as quantidades para inteiros
            quantities = np.round(quantities).astype(int)
            
            # Atualiza o DataFrame com as quantidades
            df = market_df.copy()
            df['quantidade'] = quantities
            
            # Separa os ativos em CALL e PUT
            call_df = df[df['operacao'] == 'CALL']
            put_df  = df[df['operacao'] == 'PUT']
            
            # Calcula o valor de entrada (arrecadação)
            valor_entrada_op = calculate_entry(call_df, put_df)
            
            # Penalização para arrecadação > 0
            penalty_entry = 0
            if valor_entrada_op > 0:
                penalty_entry = penalty_entry_multiplier * valor_entrada_op
            
            # Listas de strikes e quantidades
            call_strikes = call_df['strike'].tolist()
            call_qtd     = call_df['quantidade'].tolist()
            put_strikes  = put_df['strike'].tolist()
            put_qtd      = put_df['quantidade'].tolist()
            
            # Cria os valores de strike para avaliação
            strike_values = np.linspace(strike_min, strike_max, num_points)
            
            # Calcula os pesos com a normal centrada no strike central
            weights = [calcula_pesos(s, strike_central, 200) for s in strike_values]
            
            # Calcula os retornos para cada strike do intervalo
            rt_values = np.array([
                calculate_rt(s, valor_entrada_op, call_strikes, call_qtd, put_strikes, put_qtd)
                for s in strike_values
            ])
            
            # Retorno ponderado (integral do produto retorno x peso)
            weighted_return = np.trapz(rt_values * weights, strike_values)
            #weighted_return = (rt_values * weights).sum()
            
            # Retorno no strike central
            rt_center = calculate_rt(strike_central, valor_entrada_op, call_strikes, call_qtd, put_strikes, put_qtd)
            
            # Penaliza se o retorno no strike central não for o máximo
            penalty = 0
            if rt_center < rt_values.max():
                penalty = (rt_values.max() - rt_center)
            
            # Combina as métricas e subtrai as penalizações
            score = alpha * rt_center + beta * weighted_return - penalty_multiplier * penalty - penalty_entry
            
            # Como differential_evolution minimiza, retornamos o negativo do score
            return -score

        # Limites para cada quantidade (entre -10000 e 10000)
        bounds = [(-100000, 100000)] * len(market_df)
        
        with st.spinner("Calculando quantidades...", show_time=True):
            # Otimização usando Differential Evolution
            result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=1000,
                                            popsize=15, tol=0.01, workers=1)

        # Converte as melhores quantidades para inteiros
        best_quantities = np.round(result.x/10).astype(int)

        market_df['quantidade'] = best_quantities

        call_df = market_df[market_df['operacao'] == 'CALL'].copy()
        put_df = market_df[market_df['operacao'] == 'PUT'].copy()

        call_df.drop(columns=['operacao'], inplace=True)
        put_df.drop(columns=['operacao'], inplace=True)

        #Teste inicial das combinacoes com intervalo refinado no range
        min_cenario = strike_min
        max_cenario = strike_max
        granularidade = 0.01
        p_cenarios = np.arange(min_cenario, max_cenario+granularidade, granularidade)

        p_results = [calculate_rt(p, calculate_entry(call_df, put_df), 
                                call_df.strike.values, call_df.quantidade.values, 
                                put_df.strike.values, put_df.quantidade.values) 
                                for p in p_cenarios]

        arr_results = np.array(p_results)
        cobertura_absoluta = len(arr_results[arr_results > 0])
        cobertura_percentual = cobertura_absoluta/len(arr_results)

        # Example data
        x = np.array(p_cenarios)
        y = np.array(p_results)

        # Create a scatter plot
        fig = go.Figure()

        # Add positive values in green with smaller points
        fig.add_trace(go.Scatter(x=x[y >= 0], y=y[y >= 0], mode='markers', marker=dict(color='green', size=3), name='Positive'))

        # Add negative values in red with smaller points
        fig.add_trace(go.Scatter(x=x[y < 0], y=y[y < 0], mode='markers', marker=dict(color='red', size=3), name='Negative'))

        # Add a black straight line through y = 0
        fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[0, 0], mode='lines', line=dict(color='black', width=2), name='y=0'))

        # Add labels and title
        fig.update_layout(
            title='Retornos nos strikes para as quantidades calculadas',
            xaxis_title='Valor Strike',
            yaxis_title='RT'
        )

        # Show the plot
        st.plotly_chart(fig)

        arrecadacao = str(calculate_entry(call_df, put_df))

        col1, col2, col3 = st.columns(3)
        with col1:
            pass
        with col2:
            st.text("Arredadação = " + arrecadacao)
        with col3:
            pass    

        call_df['strike'] = call_df['strike'].astype(str)
        call_df['valor'] = call_df['valor'].astype(str)
        call_df['quantidade'] = call_df['quantidade'].astype(str)
        put_df['strike'] = put_df['strike'].astype(str)
        put_df['valor'] = put_df['valor'].astype(str)
        put_df['quantidade'] = put_df['quantidade'].astype(str)

        call_df['strike'] = call_df['strike'].apply(lambda x: x.replace('.', ','))
        call_df['valor'] = call_df['valor'].apply(lambda x: x.replace('.', ','))
        call_df['quantidade'] = call_df['quantidade'].apply(lambda x: x.replace(',', ''))
        put_df['strike'] = put_df['strike'].apply(lambda x: x.replace('.', ','))
        put_df['valor'] = put_df['valor'].apply(lambda x: x.replace('.', ','))
        put_df['quantidade'] = put_df['quantidade'].apply(lambda x: x.replace(',', ''))

        col1, col2 = st.columns(2)
        with col1:
            st.header("Call")
            st.dataframe(call_df)
            

        with col2:
            st.header("Put")
            st.dataframe(put_df)

        st.write(put_df['quantidade'].dtypes)


#####################################################################

elif option == "Otimização Gaussiana":

    col1, col2, col3 = st.columns(3)
    
    with col1:
        left_input = st.text_input("Strike Minimo", "113")

    with col2:
        center_input = st.text_input("Strike Central", "123")

    with col3:
        right_input = st.text_input("Strike Maximo", "133")
    
    col1, col2, col3 , col4, col5, col6, col7 = st.columns(7)
    if col4.button("Calcular", type="primary"):

        call_df = edit_call_df.copy()
        put_df = edit_put_df.copy()

        call_df['operacao'] = "CALL"
        put_df['operacao'] = "PUT"

        
        call_df['strike'] = call_df['strike'].apply(lambda x: x.replace(',', '.'))
        call_df['valor'] = call_df['valor'].apply(lambda x: x.replace(',', '.'))
        put_df['strike'] = put_df['strike'].apply(lambda x: x.replace(',', '.'))
        put_df['valor'] = put_df['valor'].apply(lambda x: x.replace(',', '.'))

        call_df['strike'] = call_df['strike'].astype(float)
        call_df['valor'] = call_df['valor'].astype(float)
        put_df['strike'] = put_df['strike'].astype(float)
        put_df['valor'] = put_df['valor'].astype(float)

        market_df = pd.concat([call_df, put_df])

        strike_central = abs(float(center_input.replace(',', '.'))) 
        strike_min = abs(float(left_input.replace(',', '.')))
        strike_max = abs(float(right_input.replace(',', '.')))  
        num_points = 50

        def objective(quantities): 
            sigma = 1            # Largura da curva da normal (quanto maior, mais larga
            alpha=1.0            # Peso para o retorno no strike central, 
            beta=1.0             # Peso para o retorno ponderado, 
            penalty_multiplier=1000      # Multiplicador para penalizar se o retorno central não for máximo
            penalty_entry_multiplier=1000  # Multiplicador para penalizar se a arrecadação for > 0):
            
           # Arredonda as quantidades para inteiros
            quantities = np.round(quantities).astype(int)
            
            # Atualiza o DataFrame com as quantidades
            df = market_df.copy()
            df['quantidade'] = quantities
            
            # Separa os ativos em CALL e PUT
            call_df = df[df['operacao'] == 'CALL']
            put_df  = df[df['operacao'] == 'PUT']
            
            # Calcula o valor de entrada (arrecadação)
            valor_entrada_op = calculate_entry(call_df, put_df)
            
            # Penalização para arrecadação > 0
            penalty_entry = 0
            if valor_entrada_op > 0:
                penalty_entry = penalty_entry_multiplier * valor_entrada_op
            
            # Listas de strikes e quantidades
            call_strikes = call_df['strike'].tolist()
            call_qtd     = call_df['quantidade'].tolist()
            put_strikes  = put_df['strike'].tolist()
            put_qtd      = put_df['quantidade'].tolist()
            
            # Cria os valores de strike para avaliação
            strike_values = np.linspace(strike_min, strike_max, num_points)
            
            # Calcula os pesos com a normal centrada no strike central
            weights = norm.pdf(strike_values, loc=strike_central, scale=sigma)
            
            # Calcula os retornos para cada strike do intervalo
            rt_values = np.array([
                calculate_rt(s, valor_entrada_op, call_strikes, call_qtd, put_strikes, put_qtd)
                for s in strike_values
            ])
            
            # Retorno ponderado (integral do produto retorno x peso)
            weighted_return = np.trapz(rt_values * weights, strike_values)
            
            # Retorno no strike central
            rt_center = calculate_rt(strike_central, valor_entrada_op, call_strikes, call_qtd, put_strikes, put_qtd)
            
            # Penaliza se o retorno no strike central não for o máximo
            penalty = 0
            if rt_center < rt_values.max():
                penalty = (rt_values.max() - rt_center)
            
            # Combina as métricas e subtrai as penalizações
            score = alpha * rt_center + beta * weighted_return - penalty_multiplier * penalty - penalty_entry
            
            # Como differential_evolution minimiza, retornamos o negativo do score
            return -score

        # Limites para cada quantidade (entre -10000 e 10000)
        bounds = [(-100000, 100000)] * len(market_df)
        
        with st.spinner("Calculando quantidades...", show_time=True):
            # Otimização usando Differential Evolution
            result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, workers=1)

        # Converte as melhores quantidades para inteiros
        best_quantities = np.round(result.x/10).astype(int)

        market_df['quantidade'] = best_quantities

        call_df = market_df[market_df['operacao'] == 'CALL'].copy()
        put_df = market_df[market_df['operacao'] == 'PUT'].copy()

        call_df.drop(columns=['operacao'], inplace=True)
        put_df.drop(columns=['operacao'], inplace=True)

        #Teste inicial das combinacoes com intervalo refinado no range
        min_cenario = strike_min
        max_cenario = strike_max
        granularidade = 0.01
        p_cenarios = np.arange(min_cenario, max_cenario+granularidade, granularidade)

        p_results = [calculate_rt(p, calculate_entry(call_df, put_df), 
                                call_df.strike.values, call_df.quantidade.values, 
                                put_df.strike.values, put_df.quantidade.values) 
                                for p in p_cenarios]

        arr_results = np.array(p_results)
        cobertura_absoluta = len(arr_results[arr_results > 0])
        cobertura_percentual = cobertura_absoluta/len(arr_results)

        # Example data
        x = np.array(p_cenarios)
        y = np.array(p_results)

        # Create a scatter plot
        fig = go.Figure()

        # Add positive values in green with smaller points
        fig.add_trace(go.Scatter(x=x[y >= 0], y=y[y >= 0], mode='markers', marker=dict(color='green', size=3), name='Positive'))

        # Add negative values in red with smaller points
        fig.add_trace(go.Scatter(x=x[y < 0], y=y[y < 0], mode='markers', marker=dict(color='red', size=3), name='Negative'))

        # Add a black straight line through y = 0
        fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[0, 0], mode='lines', line=dict(color='black', width=2), name='y=0'))

        # Add labels and title
        fig.update_layout(
            title='Retornos nos strikes para as quantidades calculadas',
            xaxis_title='Valor Strike',
            yaxis_title='RT'
        )

        # Show the plot
        st.plotly_chart(fig)

        arrecadacao = str(calculate_entry(call_df, put_df))

        col1, col2, col3 = st.columns(3)
        with col1:
            pass
        with col2:
            st.text("Arredadação = " + arrecadacao)
        with col3:
            pass    

        call_df['strike'] = call_df['strike'].astype(str)
        call_df['valor'] = call_df['valor'].astype(str)
        call_df['quantidade'] = call_df['quantidade'].astype(str)
        put_df['strike'] = put_df['strike'].astype(str)
        put_df['valor'] = put_df['valor'].astype(str)
        put_df['quantidade'] = put_df['quantidade'].astype(str)

        call_df['strike'] = call_df['strike'].apply(lambda x: x.replace('.', ','))
        call_df['valor'] = call_df['valor'].apply(lambda x: x.replace('.', ','))
        call_df['quantidade'] = call_df['quantidade'].apply(lambda x: x.replace(',', ''))
        put_df['strike'] = put_df['strike'].apply(lambda x: x.replace('.', ','))
        put_df['valor'] = put_df['valor'].apply(lambda x: x.replace('.', ','))
        put_df['quantidade'] = put_df['quantidade'].apply(lambda x: x.replace(',', ''))

        col1, col2 = st.columns(2)
        with col1:
            st.header("Call")
            st.dataframe(call_df)
            

        with col2:
            st.header("Put")
            st.dataframe(put_df)


#####################################################################

elif option == "Otimização Linear":

    col1, col2 = st.columns(2)
    
    with col1:
        left_input = st.text_input("Strike Minimo", "112")

    with col2:
        right_input = st.text_input("Strike Maximo", "135")
    
    col1, col2, col3 , col4, col5, col6, col7 = st.columns(7)
    if col4.button("Calcular", type="primary"):

        call_df = edit_call_df.copy()
        put_df = edit_put_df.copy()

        call_df['operacao'] = "CALL"
        put_df['operacao'] = "PUT"

        
        call_df['strike'] = call_df['strike'].apply(lambda x: x.replace(',', '.'))
        call_df['valor'] = call_df['valor'].apply(lambda x: x.replace(',', '.'))
        put_df['strike'] = put_df['strike'].apply(lambda x: x.replace(',', '.'))
        put_df['valor'] = put_df['valor'].apply(lambda x: x.replace(',', '.'))

        call_df['strike'] = call_df['strike'].astype(float)
        call_df['valor'] = call_df['valor'].astype(float)
        put_df['strike'] = put_df['strike'].astype(float)
        put_df['valor'] = put_df['valor'].astype(float)

        market_df = pd.concat([call_df, put_df]).reset_index(drop=True)

        strike_min = abs(float(left_input.replace(',', '.')))
        strike_max = abs(float(right_input.replace(',', '.')))  

        # Definindo o intervalo de strike_regua (exemplo: de 20 a 50)
        strike_regua_values = list(np.arange(strike_min, strike_max+0.01, 0.01))

        # Parâmetros para a técnica big-M e epsilon
        M = 1000      # Um número grande o suficiente
        epsilon = 0.001  # Valor pequeno para garantir que, quando y_s = 1, RT(s) seja estritamente positivo

        #with st.spinner("Calculando quantidades...", show_time=True):
        # Criação do modelo de otimização
        model = pl.LpProblem("Maximizar_Retorno_Combinado", pl.LpMaximize)

        # Variáveis de decisão para as quantidades de cada opção (limite de -10 a 10, por exemplo)
        quantidades = {}
        for idx, row in market_df.iterrows():
            quantidades[idx] = pl.LpVariable(f"x_{idx}", lowBound=-10000, upBound=10000, cat='Integer')

        # Restrição: o valor da entrada (arrecadação) deve ser <= 0
        model += pl.lpSum([row['valor'] * quantidades[idx] for idx, row in market_df.iterrows()]) <= 0, "Restricao_valor_entrada"
        
        # Variáveis binárias para cada valor de strike_regua no intervalo
        y_vars = {}
        # Armazenamos também as expressões dos retornos RT(s)
        expr_dict = {}

        for s in strike_regua_values:
            y_vars[s] = pl.LpVariable(f"y_{s}", cat='Binary')
            expr_s = 0
            # Para cada linha, calcula o "coeficiente" que depende do valor s
            for idx, row in market_df.iterrows():
                if row['operacao'] == 'CALL':
                    coeff = max(s - row['strike'], 0) - row['valor']
                elif row['operacao'] == 'PUT':
                    coeff = max(row['strike'] - s, 0) - row['valor']
                expr_s += coeff * quantidades[idx]
            expr_dict[s] = expr_s
            # Liga a variável binária y_s com o retorno RT(s)
            # Se y_s = 0 => RT(s) deve ser <= 0; se y_s = 1 => RT(s) >= epsilon
            model += expr_s <= M * y_vars[s], f"Link_y_upper_{s}"
            model += expr_s >= epsilon * y_vars[s], f"Link_y_lower_{s}"

        # Parâmetro de ponderação para o retorno total na função objetivo
        lambda_val = 0.001  # Ajuste esse valor conforme a escala desejada

        # Objetivo: maximizar a soma do número de cenários positivos e a soma ponderada dos retornos
        objective_expr = pl.lpSum([ y_vars[s] + lambda_val * expr_dict[s] for s in strike_regua_values ])
        model += objective_expr, "Objetivo_Combinado"

        # Resolve o modelo
        status = model.solve()

        # Atualiza as quantidades no DataFrame com os valores otimizados
        for idx, var in quantidades.items():
            market_df.loc[idx, 'quantidade'] = var.varValue

        market_df['quantidade'] = market_df['quantidade'].astype(int)
        call_df = market_df[market_df['operacao'] == 'CALL'].copy()
        put_df = market_df[market_df['operacao'] == 'PUT'].copy()

        call_df['quantidade'].fillna(0, inplace=True)
        put_df['quantidade'].fillna(0, inplace=True)
        market_df['quantidade'].fillna(0, inplace=True)

        call_df.drop(columns=['operacao'], inplace=True)
        put_df.drop(columns=['operacao'], inplace=True)

        #Teste inicial das combinacoes com intervalo refinado no range
        min_cenario = strike_min
        max_cenario = strike_max
        granularidade = 0.01
        p_cenarios = np.arange(min_cenario, max_cenario+granularidade, granularidade)

        p_results = [calculate_rt(p, calculate_entry(call_df, put_df), 
                                call_df.strike.values, call_df.quantidade.values, 
                                put_df.strike.values, put_df.quantidade.values) 
                                for p in p_cenarios]

        arr_results = np.array(p_results)
        cobertura_absoluta = len(arr_results[arr_results > 0])
        cobertura_percentual = cobertura_absoluta/len(arr_results)

        # Example data
        x = np.array(p_cenarios)
        y = np.array(p_results)

        # Create a scatter plot
        fig = go.Figure()

        # Add positive values in green with smaller points
        fig.add_trace(go.Scatter(x=x[y >= 0], y=y[y >= 0], mode='markers', marker=dict(color='green', size=3), name='Positive'))

        # Add negative values in red with smaller points
        fig.add_trace(go.Scatter(x=x[y < 0], y=y[y < 0], mode='markers', marker=dict(color='red', size=3), name='Negative'))

        # Add a black straight line through y = 0
        fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[0, 0], mode='lines', line=dict(color='black', width=2), name='y=0'))

        # Add labels and title
        fig.update_layout(
            title='Retornos nos strikes para as quantidades calculadas',
            xaxis_title='Valor Strike',
            yaxis_title='RT'
        )

        # Show the plot
        st.plotly_chart(fig)

        arrecadacao = str(calculate_entry(call_df, put_df))

        col1, col2, col3 = st.columns(3)
        with col1:
            pass
        with col2:
            st.text("Arredadação = " + arrecadacao)
        with col3:
            pass    

        call_df['strike'] = call_df['strike'].astype(str)
        call_df['valor'] = call_df['valor'].astype(str)
        call_df['quantidade'] = call_df['quantidade'].astype(str)
        put_df['strike'] = put_df['strike'].astype(str)
        put_df['valor'] = put_df['valor'].astype(str)
        put_df['quantidade'] = put_df['quantidade'].astype(str)

        call_df['strike'] = call_df['strike'].apply(lambda x: x.replace('.', ','))
        call_df['valor'] = call_df['valor'].apply(lambda x: x.replace('.', ','))
        call_df['quantidade'] = call_df['quantidade'].apply(lambda x: x.replace(',', ''))
        put_df['strike'] = put_df['strike'].apply(lambda x: x.replace('.', ','))
        put_df['valor'] = put_df['valor'].apply(lambda x: x.replace('.', ','))
        put_df['quantidade'] = put_df['quantidade'].apply(lambda x: x.replace(',', ''))

        col1, col2 = st.columns(2)
        with col1:
            st.header("Call")
            st.dataframe(call_df)
            

        with col2:
            st.header("Put")
            st.dataframe(put_df)



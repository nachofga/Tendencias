# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from scipy.stats import spearmanr, chi2
# import statsmodels.api as sm
# from statsmodels.miscmodels.ordinal_model import OrderedModel

# # Configuración de la página
# st.set_page_config(
#     page_title="En Busca de la Felicidad",
#     page_icon="😄",
#     layout="wide"
# )

# # Función para cargar y preparar los datos
# @st.cache_data
# def load_data():
#     # Cargar datos del Excel real
#     MISSING = [-1, -2, -4, -5]  # Códigos para valores faltantes
    
#     # Cargar el Excel
#     df = pd.read_excel("WVS_subset_felicidad.xlsx", engine='openpyxl')
    
#     # Reemplazar valores faltantes
#     df.replace(MISSING, np.nan, inplace=True)
    
#     # Recodificaciones siguiendo el código original
#     df['happiness'] = 5 - df['Q46']  # Invertir felicidad
    
#     # Renombrar columnas
#     df.rename(columns={
#         'Q262': 'age', 'Q50': 'fin_sat', 'Q49': 'life_sat',
#         'Q164': 'god_imp'
#     }, inplace=True)
    
#     # Invertir escalas 1-4
#     df['imp_family'] = df['Q1'].apply(lambda x: 5-x if pd.notnull(x) else x)
#     df['imp_friends'] = df['Q2'].apply(lambda x: 5-x if pd.notnull(x) else x)
#     df['sec_rec'] = df['Q131'].apply(lambda x: 5-x if pd.notnull(x) else x)
#     df['health'] = df['Q47'].apply(lambda x: 6-x if pd.notnull(x) else x)
#     df['control'] = df['Q110'].apply(lambda x: 11-x if pd.notnull(x) else x)
    
#     # Binarias/categóricas
#     df['female'] = (df['Q260']==2).astype(float)
#     df['soc_class'] = df['Q287'].astype(float)
#     df['victim'] = (df['Q144']==1).astype(float)
    
#     # Índice compuesto de crimen (promedio de 7 variables)
#     df['robos'] = df['Q132'].apply(lambda x: 5-x if pd.notnull(x) else x)
#     df['alcohol'] = df['Q133'].apply(lambda x: 5-x if pd.notnull(x) else x)
#     df['abuso_policial'] = df['Q134'].apply(lambda x: 5-x if pd.notnull(x) else x)
#     df['racismo'] = df['Q135'].apply(lambda x: 5-x if pd.notnull(x) else x)
#     df['drogas'] = df['Q136'].apply(lambda x: 5-x if pd.notnull(x) else x)
#     df['violencia'] = df['Q137'].apply(lambda x: 5-x if pd.notnull(x) else x)
#     df['acoso_sexual'] = df['Q138'].apply(lambda x: 5-x if pd.notnull(x) else x)
#     CRIME_COLS = ['robos','alcohol','abuso_policial','racismo','drogas','violencia','acoso_sexual']
#     df['indice_criminalidad'] = df[CRIME_COLS].mean(axis=1)
    
#     # Índice compuesto de importancia de relaciones
#     df['rel_import'] = df[['imp_family', 'imp_friends']].mean(axis=1)
    
#     # Variables para el análisis
#     cont_vars = ['happiness', 'age', 'fin_sat', 'health', 'life_sat',
#                 'god_imp', 'control', 'imp_family', 'imp_friends', 'rel_import', 'sec_rec', 'indice_criminalidad']
#     cat_vars = ['female', 'soc_class', 'victim']
    
#     # Listas para el análisis
#     model_vars = ['age', 'female', 'soc_class', 'fin_sat', 'health', 'life_sat',
#                  'god_imp', 'control', 'rel_import', 'sec_rec', 'indice_criminalidad', 'victim']
    
#     return df, cont_vars, cat_vars, model_vars

# # Función para calcular correlaciones con la felicidad
# def calculate_correlations(df, vars_list):
#     corr_results = []
#     for var in vars_list:
#         if var != 'happiness':
#             # CORRECCIÓN: Crear un DataFrame temporal con ambas variables
#             # y eliminar filas donde cualquiera de las dos tenga valores faltantes
#             temp_df = df[[var, 'happiness']].dropna()
            
#             # Calcular correlación solo si hay suficientes datos
#             if len(temp_df) > 5:  # Asegurar que hay suficientes datos para correlación
#                 rho, p = spearmanr(temp_df[var], temp_df['happiness'], nan_policy='omit')
#                 corr_results.append({
#                     'Variable': var,
#                     'Correlación (ρ)': rho,
#                     'p-valor': p,
#                     'Significativo': p < 0.05
#                 })
#             else:
#                 corr_results.append({
#                     'Variable': var,
#                     'Correlación (ρ)': np.nan,
#                     'p-valor': np.nan,
#                     'Significativo': False
#                 })
#     return pd.DataFrame(corr_results)

# # Función para ajustar el modelo ordinal
# def fit_ordinal_model(df, predictors):
#     # Eliminar filas con valores faltantes
#     model_df = df[predictors + ['happiness']].dropna()
    
#     X = model_df[predictors]
#     y = model_df['happiness']
    
#     mod = OrderedModel(y, X, distr='logit')
#     res = mod.fit(method='bfgs', disp=False)
    
#     # Extraer coeficientes y p-valores
#     results_df = pd.DataFrame({
#         'Variable': res.params.index,
#         'Coeficiente': res.params.values,
#         'Error Estándar': res.bse.values,
#         'p-valor': res.pvalues.values,
#         'Significativo': res.pvalues < 0.05
#     })
    
#     return res, results_df

# # Función para generar predicciones marginales
# def generate_marginal_predictions(model, result, base_values, var_to_vary, range_min, range_max, num_points=5):
#     # Crear valores base
#     base = pd.DataFrame([base_values])
    
#     # Generar secuencia de valores para la variable a variar
#     var_values = np.linspace(range_min, range_max, num_points)
    
#     # Generar predicciones
#     preds = []
#     for val in var_values:
#         # Copiar valores base y modificar la variable a variar
#         temp_base = base.copy()
#         temp_base[var_to_vary] = val
        
#         # Obtener predicciones
#         pred = model.predict(result.params, temp_base)
#         preds.append(pred[0])
    
#     # Convertir a array
#     preds = np.array(preds)
    
#     return var_values, preds

# # Cargar datos
# df, cont_vars, cat_vars, model_vars = load_data()

# # TÍTULO DEL DASHBOARD
# st.title('📊 Determinantes de la Felicidad: Dashboard Interactivo')
# st.markdown("""
# Este dashboard permite explorar los factores que influyen en la felicidad según el análisis realizado.
# Puedes interactuar con los gráficos, ajustar variables y ver cómo cambian las predicciones del modelo.
# """)

# # TABS PARA ORGANIZAR EL CONTENIDO
# tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Exploratorio", "🔄 Bivariado", "📊 Modelo", "🔮 Simulador", "📉 Interacciones"])

# # TAB 1: ANÁLISIS EXPLORATORIO
# with tab1:
#     st.header("Exploración de Variables")
#     st.markdown("Observemos la distribución de las principales variables en nuestro análisis.")
    
#     # Distribución de la felicidad
#     st.subheader("Distribución de la Variable Principal: Felicidad")
#     fig_happiness = px.histogram(df, x='happiness', nbins=4, 
#                             color_discrete_sequence=['#3498db'],
#                             labels={'happiness': 'Nivel de Felicidad'},
#                             title='Distribución de los Niveles de Felicidad')
#     fig_happiness.update_layout(xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4], 
#                                     ticktext=['Muy Infeliz', 'Infeliz', 'Feliz', 'Muy Feliz']))
#     st.plotly_chart(fig_happiness, use_container_width=True)
    
#     # Selector para variables continuas
#     st.subheader("Explorar Distribución de Variables")
#     col1, col2 = st.columns(2)
    
#     # Selección de variables para visualizar
#     with col1:
#         selected_cont_var = st.selectbox("Selecciona una variable continua:", 
#                              [v for v in cont_vars if v != 'happiness'],
#                              key="explore_cont_var") 
        
#         # Histograma para la variable seleccionada
#         fig_cont = px.histogram(df, x=selected_cont_var, 
#                            color_discrete_sequence=['#2ecc71'],
#                            title=f'Distribución de {selected_cont_var}')
#         st.plotly_chart(fig_cont, use_container_width=True)
    
#     with col2:
#         selected_cat_var = st.selectbox("Selecciona una variable categórica:", 
#                              cat_vars,
#                              key="explore_cat_var")
        
#         # Gráfico de barras para variables categóricas
#         cat_counts = df[selected_cat_var].value_counts().reset_index()
#         cat_counts.columns = [selected_cat_var, 'Count']
        
#         fig_cat = px.bar(cat_counts, x=selected_cat_var, y='Count',
#                       color_discrete_sequence=['#e74c3c'],
#                       title=f'Distribución de {selected_cat_var}')
        
#         # Etiquetas para variables específicas
#         if selected_cat_var == 'female':
#             fig_cat.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], 
#                                          ticktext=['Hombre', 'Mujer']))
#         elif selected_cat_var == 'victim':
#             fig_cat.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], 
#                                          ticktext=['No', 'Sí']))
        
#         st.plotly_chart(fig_cat, use_container_width=True)
    
#     # Mostrar estadísticas descriptivas
#     with st.expander("Ver Estadísticas Descriptivas"):
#         st.subheader("Estadísticas Descriptivas")
#         st.dataframe(df[cont_vars].describe())

# # TAB 2: ANÁLISIS BIVARIADO
# with tab2:
#     st.header("Relaciones Bivariadas con la Felicidad")
#     st.markdown("Explora cómo cada variable se relaciona con los niveles de felicidad.")
    
#     # Seleccionar tipo de variable para analizar
#     var_type = st.radio("Tipo de variable a analizar:", 
#                    ["Continua", "Categórica"],
#                    key="bivariate_var_type")
    
#     if var_type == "Continua":
#         # Seleccionar variable continua
#         selected_var = st.selectbox("Selecciona una variable continua:", 
#                         [v for v in cont_vars if v != 'happiness'],
#                         key="bivariate_cont_var") 
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # CORRECCIÓN: Filtrar juntas las filas con valores faltantes
#             scatter_df = df[[selected_var, 'happiness']].dropna()
            
#             # Gráfico de dispersión
#             fig_scatter = px.scatter(scatter_df, x=selected_var, y='happiness', 
#                                  color_discrete_sequence=['#3498db'],
#                                  opacity=0.6,
#                                  labels={'happiness': 'Nivel de Felicidad'},
#                                  title=f'Relación entre {selected_var} y Felicidad')
            
#             # Añadir línea de tendencia
#             fig_scatter.update_layout(xaxis_title=selected_var, yaxis_title='Felicidad')
#             fig_scatter.update_traces(marker=dict(size=8))
            
#             # Añadir línea de tendencia solo si hay suficientes datos
#             if len(scatter_df) > 2:
#                 z = np.polyfit(scatter_df[selected_var], scatter_df['happiness'], 1)
#                 p = np.poly1d(z)
                
#                 x_range = np.linspace(scatter_df[selected_var].min(), scatter_df[selected_var].max(), 100)
#                 fig_scatter.add_trace(go.Scatter(
#                     x=x_range,
#                     y=p(x_range),
#                     mode='lines',
#                     name='Tendencia',
#                     line=dict(color='red', width=2)
#                 ))
            
#             st.plotly_chart(fig_scatter, use_container_width=True)
        
#         with col2:
#             # CORRECCIÓN: Usar el mismo DataFrame filtrado
#             if len(scatter_df) > 5:  # Asegurarse de que hay suficientes datos
#                 # Calcular estadísticas
#                 rho, p = spearmanr(scatter_df[selected_var], scatter_df['happiness'])
                
#                 # Mostrar correlación
#                 st.metric("Correlación de Spearman (ρ)", f"{rho:.3f}")
#                 st.metric("p-valor", f"{p:.3f}", delta="Significativo" if p < 0.05 else "No significativo")
#             else:
#                 st.warning("No hay suficientes datos válidos para calcular correlaciones.")
            
#             # Boxplot por nivel de felicidad usando el mismo DataFrame
#             if len(scatter_df) > 5:
#                 fig_box = px.box(scatter_df, x='happiness', y=selected_var, 
#                              color_discrete_sequence=['#2ecc71'],
#                              title=f'Distribución de {selected_var} por Nivel de Felicidad')
                
#                 fig_box.update_layout(xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4], 
#                                          ticktext=['Muy Infeliz', 'Infeliz', 'Feliz', 'Muy Feliz']))
                
#                 st.plotly_chart(fig_box, use_container_width=True)
#             else:
#                 st.info("No hay suficientes datos para mostrar el boxplot.")
    
#     else:  # Categórica
#         # Seleccionar variable categórica
#         selected_var = st.selectbox("Selecciona una variable categórica:", 
#                         cat_vars,
#                         key="bivariate_cat_var") 
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # CORRECCIÓN: Filtrar juntas las filas con valores faltantes
#             crosstab_df = df[[selected_var, 'happiness']].dropna()
            
#             if len(crosstab_df) > 10:  # Asegurar que hay suficientes datos
#                 # Gráfico de barras apiladas
#                 # Crear tabla de contingencia
#                 crosstab = pd.crosstab(crosstab_df[selected_var], crosstab_df['happiness'], normalize='index') * 100
#                 crosstab = crosstab.reset_index()
                
#                 # Convertir a formato largo para Plotly
#                 crosstab_long = pd.melt(crosstab, id_vars=[selected_var], 
#                                     var_name='happiness', value_name='percentage')
                
#                 # Gráfico de barras apiladas
#                 fig_stacked = px.bar(crosstab_long, x=selected_var, y='percentage', color='happiness',
#                                  barmode='stack',
#                                  labels={'percentage': 'Porcentaje (%)'},
#                                  title=f'Proporción de Felicidad por {selected_var}')
                
#                 # Etiquetas personalizadas
#                 if selected_var == 'female':
#                     fig_stacked.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], 
#                                                      ticktext=['Hombre', 'Mujer']))
#                 elif selected_var == 'victim':
#                     fig_stacked.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], 
#                                                      ticktext=['No', 'Sí']))
                
#                 # Actualizar colores para que coincidan con el nivel de felicidad
#                 fig_stacked.update_layout(coloraxis=dict(colorscale=['red', 'orange', 'lightgreen', 'darkgreen']))
                
#                 st.plotly_chart(fig_stacked, use_container_width=True)
#             else:
#                 st.warning("No hay suficientes datos para generar el gráfico de barras apiladas.")
        
#         with col2:
#             # Usar el mismo DataFrame filtrado para todas las operaciones
#             if len(crosstab_df) > 10:
#                 # Tabla de contingencia
#                 st.subheader("Tabla de Contingencia")
#                 contingency_table = pd.crosstab(crosstab_df[selected_var], crosstab_df['happiness'])
#                 st.dataframe(contingency_table)
                
#                 # Prueba estadística
#                 if crosstab_df[selected_var].nunique() == 2:
#                     # Para variables binarias usamos Mann-Whitney (equivalente)
#                     from scipy.stats import mannwhitneyu
                    
#                     g0 = crosstab_df[crosstab_df[selected_var]==0]['happiness']
#                     g1 = crosstab_df[crosstab_df[selected_var]==1]['happiness']
                    
#                     # Solo calcular si ambos grupos tienen datos
#                     if len(g0) > 0 and len(g1) > 0:
#                         U, p_val = mannwhitneyu(g0, g1)
#                         test_name = "Mann-Whitney U"
#                         test_stat = U
#                     else:
#                         p_val = np.nan
#                         test_name = "No hay suficientes datos"
#                         test_stat = np.nan
#                 else:
#                     # Para variables con más categorías usamos Kruskal-Wallis
#                     from scipy.stats import kruskal
                    
#                     # Agrupar por categorías
#                     groups = []
#                     for val in crosstab_df[selected_var].unique():
#                         group = crosstab_df[crosstab_df[selected_var]==val]['happiness'].values
#                         if len(group) > 0:
#                             groups.append(group)
                    
#                     # Solo calcular si hay al menos dos grupos con datos
#                     if len(groups) >= 2:
#                         H, p_val = kruskal(*groups)
#                         test_name = "Kruskal-Wallis H"
#                         test_stat = H
#                     else:
#                         p_val = np.nan
#                         test_name = "No hay suficientes datos"
#                         test_stat = np.nan
                
#                 # Mostrar resultados de la prueba
#                 st.subheader("Prueba Estadística")
#                 if not np.isnan(p_val):
#                     st.write(f"**{test_name}:** {test_stat:.2f}")
#                     st.write(f"**p-valor:** {p_val:.3f}")
#                     st.write(f"**Resultado:** {'Significativo' if p_val < 0.05 else 'No significativo'}")
#                 else:
#                     st.info(f"No se pudo calcular la prueba estadística: {test_name}")
#             else:
#                 st.warning("No hay suficientes datos para realizar análisis estadísticos.")
    
#     # Tabla completa de correlaciones
#     with st.expander("Ver Todas las Correlaciones con Felicidad"):
#         # Calcular todas las correlaciones con felicidad
#         corr_df = calculate_correlations(df, cont_vars + cat_vars)
        
#         # Ordenar por magnitud absoluta de correlación (solo para filas con datos)
#         valid_corr = corr_df.dropna(subset=['Correlación (ρ)'])
#         if len(valid_corr) > 0:
#             valid_corr['Magnitud'] = valid_corr['Correlación (ρ)'].abs()
#             valid_corr = valid_corr.sort_values('Magnitud', ascending=False).drop('Magnitud', axis=1)
            
#             # Concatenar con filas que tienen NaN en correlación
#             nan_corr = corr_df[corr_df['Correlación (ρ)'].isna()]
#             corr_df = pd.concat([valid_corr, nan_corr]).reset_index(drop=True)
            
#             st.dataframe(corr_df)
#         else:
#             st.warning("No hay suficientes datos para calcular correlaciones.")

# # TAB 3: RESULTADOS DEL MODELO
# with tab3:
#     st.header("Modelo Multivariado")
#     st.markdown("Análisis del modelo ordinal logístico para predecir felicidad.")
    
#     # Verificar si hay suficientes datos para el modelo
#     model_data = df[model_vars + ['happiness']].dropna()
#     if len(model_data) > len(model_vars) + 10:  # Comprobar que hay suficientes observaciones
#         # Ajustar el modelo
#         try:
#             model, model_results = fit_ordinal_model(df, model_vars)
            
#             col1, col2 = st.columns([3, 2])
            
#             with col1:
#                 # Gráfico de coeficientes
#                 model_results_plot = model_results.sort_values('Coeficiente')
                
#                 fig_coef = go.Figure()
                
#                 # Añadir barras para coeficientes
#                 fig_coef.add_trace(go.Bar(
#                     y=model_results_plot['Variable'],
#                     x=model_results_plot['Coeficiente'],
#                     orientation='h',
#                     marker_color=['#e74c3c' if x < 0 else '#2ecc71' for x in model_results_plot['Coeficiente']],
#                     name='Coeficiente',
#                     error_x=dict(
#                         type='data',
#                         array=model_results_plot['Error Estándar'],
#                         visible=True
#                     )
#                 ))
                
#                 fig_coef.update_layout(
#                     title='Coeficientes del Modelo Ordinal',
#                     xaxis_title='Coeficiente (log-odds)',
#                     yaxis_title='Variable',
#                     height=600
#                 )
                
#                 st.plotly_chart(fig_coef, use_container_width=True)
            
#             with col2:
#                 # Tabla de resultados del modelo
#                 st.subheader("Coeficientes y Significancia")
                
#                 # Ordenar por significancia y magnitud absoluta
#                 display_results = model_results.copy()
#                 display_results['Magnitud'] = display_results['Coeficiente'].abs()
#                 display_results = display_results.sort_values(['Significativo', 'Magnitud'], ascending=[False, False])
                
#                 # Formatear para mostrar
#                 display_results['Coeficiente'] = display_results['Coeficiente'].round(3)
#                 display_results['Error Estándar'] = display_results['Error Estándar'].round(3)
#                 display_results['p-valor'] = display_results['p-valor'].round(3)
                
#                 # Mostrar solo las columnas relevantes
#                 st.dataframe(display_results[['Variable', 'Coeficiente', 'Error Estándar', 'p-valor', 'Significativo']])
                
#                 # Estadísticas del modelo
#                 st.subheader("Estadísticas del Modelo")
#                 st.write(f"**Log-Likelihood:** {model.llf:.2f}")
#                 st.write(f"**AIC:** {model.aic:.2f}")
#                 st.write(f"**BIC:** {model.bic:.2f}")
#                 st.write(f"**Número de observaciones:** {model.nobs}")
            
#             # Interpretación del modelo
#             st.subheader("Interpretación de los Resultados")
            
#             # Filtrar para variables con coeficientes positivos/negativos
#             pos_coef = model_results[model_results['Coeficiente'] > 0]
#             neg_coef = model_results[model_results['Coeficiente'] < 0]
            
#             # Ordenar por importancia (magnitud del coeficiente)
#             top_positive = pos_coef.sort_values('Coeficiente', ascending=False).head(3) if len(pos_coef) > 0 else pd.DataFrame()
#             top_negative = neg_coef.sort_values('Coeficiente').head(3) if len(neg_coef) > 0 else pd.DataFrame()
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.markdown("#### Factores que Aumentan la Felicidad")
#                 if len(top_positive) > 0:
#                     for _, row in top_positive.iterrows():
#                         effect_size = "fuerte" if abs(row['Coeficiente']) > 0.3 else "moderado" if abs(row['Coeficiente']) > 0.1 else "leve"
#                         st.markdown(f"- **{row['Variable']}** (β = {row['Coeficiente']:.3f}): Efecto {effect_size} positivo")
#                 else:
#                     st.info("No se encontraron factores con efecto positivo significativo.")
            
#             with col2:
#                 st.markdown("#### Factores que Disminuyen la Felicidad")
#                 if len(top_negative) > 0:
#                     for _, row in top_negative.iterrows():
#                         effect_size = "fuerte" if abs(row['Coeficiente']) > 0.3 else "moderado" if abs(row['Coeficiente']) > 0.1 else "leve"
#                         st.markdown(f"- **{row['Variable']}** (β = {row['Coeficiente']:.3f}): Efecto {effect_size} negativo")
#                 else:
#                     st.info("No se encontraron factores con efecto negativo significativo.")
            
#             # Resumen general
#             st.markdown("""
#             #### Interpretación General
            
#             El modelo ordinal logístico muestra cómo distintas variables influyen en la probabilidad de tener mayores niveles de felicidad.
            
#             - Los **coeficientes positivos** indican que un aumento en esa variable está asociado con mayor probabilidad de felicidad.
#             - Los **coeficientes negativos** indican que un aumento en esa variable está asociado con menor probabilidad de felicidad.
#             - La **magnitud** del coeficiente indica la fuerza de la asociación.
#             - El **p-valor** indica si la asociación es estadísticamente significativa (p < 0.05).
            
#             Es importante destacar que estos son efectos ajustados, es decir, el efecto de cada variable controlando por todas las demás variables del modelo.
#             """)
#         except Exception as e:
#             st.error(f"Error al ajustar el modelo: {str(e)}")
#             st.info("Esto puede deberse a problemas de convergencia o multicolinealidad en los datos. Pruebe con un conjunto diferente de variables predictoras.")
#     else:
#         st.warning(f"No hay suficientes datos para ajustar el modelo. Se necesitan al menos {len(model_vars) + 10} observaciones completas.")

# # TAB 4: SIMULADOR DE PREDICCIONES
# with tab4:
#     st.header("Simulador de Felicidad")
#     st.markdown("Ajusta las variables para ver cómo cambian las predicciones de felicidad.")
    
#     # Verificar si hay un modelo ajustado
#     try:
#         # Intentar acceder a model para ver si está definido
#         _ = model.params
        
#         # Valores por defecto (medianas)
#         default_values = {var: float(df[var].dropna().median()) for var in model_vars}
        
#         # Layout con dos columnas: controles y gráfico
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             st.subheader("Ajustar Variables")
            
#             # Crear sliders para cada variable y almacenar valores
#             adjusted_values = {}
            
#             # Organizar variables en grupos
#             demo_vars = ['age', 'female', 'soc_class']
#             wellbeing_vars = ['health', 'life_sat', 'fin_sat']
#             values_vars = ['god_imp', 'control', 'rel_import', 'sec_rec']
#             context_vars = ['indice_criminalidad', 'victim']
            
#             # Variables demográficas
#             st.markdown("##### Variables Demográficas")
            
#             age = st.slider("Edad", 
#                         int(df['age'].dropna().min()), int(df['age'].dropna().max()), 
#                         int(default_values['age']),
#                         key="sim_age")
#             adjusted_values['age'] = age
            
#             female = st.radio("Género", 
#                         options=[0, 1], 
#                         format_func=lambda x: "Mujer" if x == 1 else "Hombre",
#                         index=int(default_values['female']),
#                         key="sim_female")
#             adjusted_values['female'] = female
            
#             soc_class = st.selectbox("Clase Social", 
#                                 options=[1, 2, 3, 4, 5],
#                                 format_func=lambda x: ["Alta", "Media-Alta", "Media", "Media-Baja", "Baja"][int(x)-1],
#                                 index=int(default_values['soc_class'])-1,
#                                 key="sim_soc_class")
#             adjusted_values['soc_class'] = soc_class
            
#             # Variables de bienestar
#             st.markdown("##### Bienestar")
            
#             health = st.slider("Salud (1=Muy buena, 5=Muy mala)", 
#                         1, 5, int(default_values['health']),
#                         key="sim_health")
#             adjusted_values['health'] = health
            
#             life_sat = st.slider("Satisfacción con la Vida", 
#                             1, 10, int(default_values['life_sat']),
#                             key="sim_life_sat")
#             adjusted_values['life_sat'] = life_sat
            
#             fin_sat = st.slider("Satisfacción Financiera", 
#                             1, 10, int(default_values['fin_sat']),
#                             key="sim_fin_sat")
#             adjusted_values['fin_sat'] = fin_sat
            
#             # Variables de valores
#             with st.expander("Valores y Actitudes"):
#                 god_imp = st.slider("Importancia de Dios", 
#                                 1, 10, int(default_values['god_imp']),
#                                 key="sim_god_imp")
#                 adjusted_values['god_imp'] = god_imp
                
#                 control = st.slider("Control sobre la Vida", 
#                                 1, 10, int(default_values['control']),
#                                 key="sim_control")
#                 adjusted_values['control'] = control
                
#                 rel_import = st.slider("Importancia de Relaciones", 
#                                 1.0, 4.0, float(default_values['rel_import']), 0.1,
#                                 key="sim_rel_import")
#                 adjusted_values['rel_import'] = rel_import
                
#                 sec_rec = st.slider("Seguridad Residencial", 
#                                 1, 4, int(default_values['sec_rec']),
#                                 key="sim_sec_rec")
#                 adjusted_values['sec_rec'] = sec_rec
            
#             # Variables contextuales
#             with st.expander("Contexto Social"):
#                 indice_criminalidad = st.slider("Índice de Criminalidad", 
#                                 float(df['indice_criminalidad'].dropna().min()), float(df['indice_criminalidad'].dropna().max()), 
#                                 float(default_values['indice_criminalidad']), 0.1,
#                                 key="sim_indice_criminalidad")
#                 adjusted_values['indice_criminalidad'] = indice_criminalidad
                
#                 victim = st.radio("Víctima de Delito", 
#                             options=[0, 1], 
#                             format_func=lambda x: "Sí" if x == 1 else "No",
#                             index=int(default_values['victim']),
#                             key="sim_victim")
#                 adjusted_values['victim'] = victim
        
#         with col2:
#             st.subheader("Predicciones de Felicidad")
            
#             # Crear un DataFrame con los valores ajustados
#             pred_df = pd.DataFrame([adjusted_values])
            
#             try:
#                 # Generar predicciones
#                 predictions = model.model.predict(model.params, exog=pred_df)[0]
                
#                 # Crear gráfico de probabilidades
#                 labels = ['Muy Infeliz', 'Infeliz', 'Feliz', 'Muy Feliz']
#                 colors = ['#e74c3c', '#f39c12', '#2ecc71', '#27ae60']
                
#                 # Gráfico de barras horizontales
#                 fig_pred = go.Figure()
                
#                 for i, (label, prob, color) in enumerate(zip(labels, predictions, colors)):
#                     fig_pred.add_trace(go.Bar(
#                         y=[label],
#                         x=[prob * 100],
#                         orientation='h',
#                         name=label,
#                         marker_color=color,
#                         text=[f"{prob * 100:.1f}%"],
#                         textposition='auto'
#                     ))
                
#                 fig_pred.update_layout(
#                     title='Probabilidad de Cada Nivel de Felicidad',
#                     xaxis_title='Probabilidad (%)',
#                     yaxis_title='Nivel de Felicidad',
#                     xaxis=dict(range=[0, 100]),
#                     height=300,
#                     barmode='group'
#                 )
                
#                 st.plotly_chart(fig_pred, use_container_width=True)
                
#                 # Mostrar el nivel más probable
#                 max_prob_index = np.argmax(predictions)
#                 max_prob_label = labels[max_prob_index]
#                 max_prob_value = predictions[max_prob_index]
                
#                 st.markdown(f"### Resultado:")
#                 st.markdown(f"### Con estas características, la persona más probablemente sería:")
                
#                 # Estilo para el resultado
#                 result_color = colors[max_prob_index]
#                 st.markdown(
#                     f"""
#                     <div style="background-color: {result_color}; padding: 20px; border-radius: 10px; text-align: center;">
#                         <h1 style="color: white;">{max_prob_label}</h1>
#                         <h3 style="color: white;">Probabilidad: {max_prob_value*100:.1f}%</h3>
#                     </div>
#                     """, 
#                     unsafe_allow_html=True
#                 )
                
#                 # Gráfico adicional: variable focal
#                 st.subheader("Análisis de Sensibilidad")
                
#                 # Seleccionar variable focal
#                 focal_var = st.selectbox("Selecciona una variable para análisis de sensibilidad:", 
#                                     model_vars,
#                                     index=model_vars.index('fin_sat'),
#                                     key="sim_focal_var")
                
#                 # Determinar el rango para la variable focal
#                 if focal_var in ['female', 'victim']:
#                     # Para variables binarias solo hay dos valores
#                     var_min, var_max = 0, 1
#                     num_points = 2
#                 else:
#                     # Para variables continuas, usar rango real
#                     var_min = float(df[focal_var].dropna().min())
#                     var_max = float(df[focal_var].dropna().max())
#                     num_points = 10
                
#                 # Generar predicciones marginales
#                 focal_values, marginal_preds = generate_marginal_predictions(
#                     model.model, model, adjusted_values, focal_var, var_min, var_max, num_points)
                
#                 # Crear gráfico de líneas para cada nivel de felicidad
#                 fig_marginal = go.Figure()
                
#                 for i, label in enumerate(labels):
#                     fig_marginal.add_trace(go.Scatter(
#                         x=focal_values,
#                         y=marginal_preds[:, i] * 100,
#                         mode='lines+markers',
#                         name=label,
#                         line=dict(color=colors[i], width=3)
#                     ))
                
#                 # Añadir línea vertical en el valor actual
#                 fig_marginal.add_vline(
#                     x=adjusted_values[focal_var],
#                     line=dict(color="black", width=2, dash="dash"),
#                     annotation_text="Valor Actual"
#                 )
                
#                 # Actualizar layout
#                 fig_marginal.update_layout(
#                     title=f'Cómo {focal_var} afecta la Probabilidad de Felicidad',
#                     xaxis_title=focal_var,
#                     yaxis_title='Probabilidad (%)',
#                     height=400
#                 )
                
#                 st.plotly_chart(fig_marginal, use_container_width=True)
#             except Exception as e:
#                 st.error(f"Error al generar predicciones: {str(e)}")
#     except NameError:
#         st.warning("No hay un modelo ajustado disponible. Primero debe ir a la pestaña 'Modelo' para ajustar el modelo.")

# # TAB 5: ANÁLISIS DE INTERACCIONES
# with tab5:
#     st.header("Análisis de Interacciones")
#     st.markdown("Explora cómo diferentes variables interactúan entre sí para afectar la felicidad.")
    
#     # Verificar si hay un modelo ajustado
#     try:
#         # Intentar acceder a model para ver si está definido
#         _ = model.params
        
#         # Seleccionar variables para interacción
#         col1, col2 = st.columns(2)
        
#         with col1:
#             var1 = st.selectbox("Primera Variable:", 
#                         model_vars, 
#                         index=model_vars.index('health'),
#                         key="interaction_var1")
        
#         with col2:
#             # Filtrar para no seleccionar la misma variable
#             var2_options = [v for v in model_vars if v != var1]
#             var2_index = 0  # Por defecto, tomar el primer elemento
            
#             var2 = st.selectbox("Segunda Variable:", 
#                         var2_options,
#                         index=var2_index,
#                         key="interaction_var2")
                
#         # Crear nombre de interacción
#         interaction_name = f"{var1}_x_{var2}"
        
#         try:
#             # Datos para el modelo
#             interaction_df = df[model_vars + ['happiness']].dropna()
            
#             # Verificar si hay suficientes datos
#             if len(interaction_df) > len(model_vars) + 10:
#                 # Crear modelo con interacción
#                 interaction_X = interaction_df[model_vars].copy()
#                 interaction_X[interaction_name] = interaction_X[var1] * interaction_X[var2]
                
#                 interaction_mod = OrderedModel(interaction_df['happiness'], interaction_X, distr='logit')
#                 interaction_res = interaction_mod.fit(method='bfgs', disp=False)
                
#                 # Extraer coeficientes
#                 interaction_results = pd.DataFrame({
#                     'Variable': interaction_res.params.index,
#                     'Coeficiente': interaction_res.params.values,
#                     'Error Estándar': interaction_res.bse.values,
#                     'p-valor': interaction_res.pvalues.values,
#                     'Significativo': interaction_res.pvalues < 0.05
#                 })
                
#                 # Mostrar resultados de la interacción
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     # Enfocarse en las variables de interés y la interacción
#                     focus_vars = [var1, var2, interaction_name]
#                     focus_results = interaction_results[interaction_results['Variable'].isin(focus_vars)]
                    
#                     # Verificar que tenemos resultados para las variables de interés
#                     if len(focus_results) == 3:
#                         # Gráfico de coeficientes para las variables de interés
#                         fig_interaction = go.Figure()
                        
#                         # Obtener el color para la interacción
#                         interaction_coef = focus_results.loc[focus_results['Variable']==interaction_name, 'Coeficiente'].values[0]
#                         interaction_color = '#e74c3c' if interaction_coef < 0 else '#2ecc71'
                        
#                         # Añadir barras para coeficientes
#                         fig_interaction.add_trace(go.Bar(
#                             y=focus_results['Variable'],
#                             x=focus_results['Coeficiente'],
#                             orientation='h',
#                             marker_color=['#3498db', '#3498db', interaction_color],
#                             name='Coeficiente',
#                             error_x=dict(
#                                 type='data',
#                                 array=focus_results['Error Estándar'],
#                                 visible=True
#                             )
#                         ))
                        
#                         fig_interaction.update_layout(
#                             title=f'Coeficientes del Modelo con Interacción {var1} × {var2}',
#                             xaxis_title='Coeficiente (log-odds)',
#                             yaxis_title='Variable',
#                             height=300
#                         )
                        
#                         st.plotly_chart(fig_interaction, use_container_width=True)
                        
#                         # Visualizar la interacción con un mapa de calor
#                         if var1 not in ['female', 'victim'] and var2 not in ['female', 'victim']:
#                             # Crear una cuadrícula de valores para las dos variables
#                             var1_values = np.linspace(df[var1].dropna().min(), df[var1].dropna().max(), 10)
#                             var2_values = np.linspace(df[var2].dropna().min(), df[var2].dropna().max(), 10)
                            
#                             # Crear malla
#                             var1_grid, var2_grid = np.meshgrid(var1_values, var2_values)
                            
#                             # Valores para predicción
#                             grid_df = pd.DataFrame({
#                                 var1: var1_grid.flatten(),
#                                 var2: var2_grid.flatten()
#                             })
                            
#                             # Añadir valores por defecto para otras variables
#                             for var in model_vars:
#                                 if var not in [var1, var2]:
#                                     grid_df[var] = default_values[var]
                            
#                             # Añadir interacción
#                             grid_df[interaction_name] = grid_df[var1] * grid_df[var2]
                            
#                             # Hacer predicciones
#                             preds = interaction_mod.predict(interaction_res.params, exog=grid_df)
                            
#                             # Probabilidad del nivel "Muy Feliz" (nivel 4)
#                             happy_probs = preds[:, 3].reshape(10, 10) * 100
                            
#                             # Crear mapa de calor
#                             fig_heatmap = go.Figure(data=go.Heatmap(
#                                 z=happy_probs,
#                                 x=var1_values,
#                                 y=var2_values,
#                                 colorscale='Viridis',
#                                 colorbar=dict(title='Probabilidad de<br>Ser Muy Feliz (%)'),
#                                 hovertemplate='%{x:.2f}, %{y:.2f}: %{z:.1f}%<extra></extra>'
#                             ))
                            
#                             fig_heatmap.update_layout(
#                                 title=f'Probabilidad de Ser Muy Feliz según {var1} y {var2}',
#                                 xaxis_title=var1,
#                                 yaxis_title=var2,
#                                 height=500
#                             )
                            
#                             st.plotly_chart(fig_heatmap, use_container_width=True)
#                         else:
#                             # Para variables binarias, usar gráficos de líneas agrupadas
#                             # Crear categorías
#                             if var1 in ['female', 'victim']:
#                                 cat_var, num_var = var1, var2
#                             else:
#                                 cat_var, num_var = var2, var1
                            
#                             # Crear valores para cada nivel
#                             cat_levels = [0, 1]
#                             cat_labels = {
#                                 'female': ['Hombre', 'Mujer'],
#                                 'victim': ['No Víctima', 'Víctima']
#                             }.get(cat_var, ['Nivel 0', 'Nivel 1'])  # Fallback por si acaso
                            
#                             num_values = np.linspace(df[num_var].dropna().min(), df[num_var].dropna().max(), 5)
                            
#                             # Crear DataFrame para predicciones
#                             pred_data = []
                            
#                             for cat_val in cat_levels:
#                                 for num_val in num_values:
#                                     temp_dict = default_values.copy()
#                                     temp_dict[cat_var] = cat_val
#                                     temp_dict[num_var] = num_val
#                                     temp_dict[interaction_name] = cat_val * num_val
                                    
#                                     # Crear DataFrame para la predicción
#                                     pred_df = pd.DataFrame([temp_dict])
                                    
#                                     # Hacer predicción
#                                     pred = interaction_mod.predict(interaction_res.params, exog=pred_df)
                                    
#                                     # Guardar resultado
#                                     pred_data.append({
#                                         'cat_value': cat_val,
#                                         'cat_label': cat_labels[cat_levels.index(cat_val)],
#                                         'num_value': num_val,
#                                         'prob_happy': pred[0, 3] * 100  # Probabilidad de ser muy feliz
#                                     })
                            
#                             # Convertir a DataFrame
#                             pred_df = pd.DataFrame(pred_data)
                            
#                             # Crear gráfico de líneas agrupadas
#                             fig_lines = px.line(
#                                 pred_df, 
#                                 x='num_value', 
#                                 y='prob_happy',
#                                 color='cat_label',
#                                 labels={
#                                     'num_value': num_var,
#                                     'prob_happy': 'Probabilidad de Ser Muy Feliz (%)',
#                                     'cat_label': cat_var
#                                 },
#                                 title=f'Interacción entre {cat_var} y {num_var}',
#                                 color_discrete_sequence=['#3498db', '#e74c3c']
#                             )
                            
#                             fig_lines.update_layout(height=400)
                            
#                             st.plotly_chart(fig_lines, use_container_width=True)
#                     else:
#                         st.warning("No se pudieron obtener resultados completos para las variables seleccionadas.")
                
#                 with col2:
#                     # Detalles de la interacción
#                     st.subheader("Detalles de la Interacción")
                    
#                     # Verificar que la interacción existe en los resultados
#                     if interaction_name in interaction_results['Variable'].values:
#                         # Información sobre la interacción
#                         interaction_coef = interaction_results.loc[interaction_results['Variable'] == interaction_name, 'Coeficiente'].values[0]
#                         interaction_pval = interaction_results.loc[interaction_results['Variable'] == interaction_name, 'p-valor'].values[0]
#                         interaction_sig = interaction_pval < 0.05
                        
#                         st.metric("Coeficiente de Interacción", f"{interaction_coef:.3f}")
#                         st.metric("p-valor", f"{interaction_pval:.3f}", delta="Significativo" if interaction_sig else "No significativo")
                        
#                         # Interpretación de la interacción
#                         st.subheader("Interpretación")
                        
#                         if interaction_sig:
#                             if interaction_coef > 0:
#                                 st.markdown(f"""
#                                 La interacción entre **{var1}** y **{var2}** es **positiva y significativa**.
                                
#                                 Esto significa que el efecto positivo de una variable sobre la felicidad se intensifica cuando la otra variable también aumenta.
                                
#                                 En otras palabras, estas variables se **refuerzan mutuamente** en su impacto sobre la felicidad.
#                                 """)
#                             else:
#                                 st.markdown(f"""
#                                 La interacción entre **{var1}** y **{var2}** es **negativa y significativa**.
                                
#                                 Esto significa que el efecto de una variable sobre la felicidad disminuye cuando la otra variable aumenta.
                                
#                                 En otras palabras, estas variables **atenúan mutuamente** su impacto sobre la felicidad.
#                                 """)
#                         else:
#                             st.markdown(f"""
#                             La interacción entre **{var1}** y **{var2}** no es estadísticamente significativa.
                            
#                             Esto sugiere que estas variables influyen en la felicidad de manera **independiente**, sin reforzarse ni atenuarse mutuamente.
#                             """)
                        
#                         # Comparación con el modelo original
#                         st.subheader("Comparación de Modelos")
                        
#                         # Log-likelihood del modelo original y con interacción
#                         ll_original = model.llf
#                         ll_interaction = interaction_res.llf
                        
#                         # Test de razón de verosimilitud
#                         lr_test = -2 * (ll_original - ll_interaction)
#                         p_value = 1 - chi2.cdf(lr_test, df=1)
                        
#                         st.write(f"**Log-likelihood Original:** {ll_original:.2f}")
#                         st.write(f"**Log-likelihood con Interacción:** {ll_interaction:.2f}")
#                         st.write(f"**Test Razón de Verosimilitud:** {lr_test:.2f}")
#                         st.write(f"**p-valor:** {p_value:.3f}")
                        
#                         if p_value < 0.05:
#                             st.success("El modelo con interacción es significativamente mejor que el modelo original.")
#                         else:
#                             st.info("No hay evidencia suficiente de que el modelo con interacción sea mejor que el original.")
#                     else:
#                         st.warning(f"No se encontró el término de interacción {interaction_name} en los resultados del modelo.")
                
#                 # Tabla comparativa de coeficientes
#                 with st.expander("Ver Comparación Completa de Coeficientes"):
#                     # Verificar que tenemos ambos conjuntos de resultados
#                     if 'model_results' in locals() and 'interaction_results' in locals():
#                         try:
#                             # Unir resultados
#                             comparison = pd.merge(
#                                 model_results[['Variable', 'Coeficiente', 'p-valor']],
#                                 interaction_results[['Variable', 'Coeficiente', 'p-valor']],
#                                 on='Variable',
#                                 suffixes=(' (Original)', ' (Interacción)')
#                             )
                            
#                             # Formatear para mostrar
#                             comparison = comparison.round(3)
                            
#                             st.dataframe(comparison)
#                         except Exception as e:
#                             st.error(f"Error al crear la tabla comparativa: {str(e)}")
#                     else:
#                         st.warning("No se pudieron obtener los resultados de ambos modelos para hacer la comparación.")
#             else:
#                 st.warning(f"No hay suficientes datos para ajustar el modelo de interacción. Se necesitan al menos {len(model_vars) + 10} observaciones completas.")
#         except Exception as e:
#             st.error(f"Error al ajustar el modelo de interacción: {str(e)}")
#             st.info("Esto puede deberse a problemas de convergencia o multicolinealidad en los datos. Pruebe con variables diferentes o con más datos.")
#     except NameError:
#         st.warning("No hay un modelo ajustado disponible. Primero debe ir a la pestaña 'Modelo' para ajustar el modelo.")

# # PIE DE PÁGINA
# st.markdown("""---""")
# st.markdown("""
# <div style="text-align: center">
#     <p>Dashboard creado para el análisis de determinantes de la felicidad</p>
#     <p>Basado en World Values Survey</p>
#     <p>Eduardo Alzu / Diego Antón / Marcos Domínguez / Roberto García / Ignacio Fumanal / Ángel Prados</p>
# </div>
# """, unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import spearmanr, chi2, mannwhitneyu, kruskal
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import plotly.figure_factory as ff
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import time

# Configuración de la página con tema personalizado
st.set_page_config(
    page_title="En Busca de la Felicidad",
    page_icon="😄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    .highlight {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 0.5rem solid #4e8df5;
    }
    .info-box {
        background-color: #e1f5fe;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 0.3rem solid #03a9f4;
    }
    .warning-box {
        background-color: #fff8e1;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 0.3rem solid #ffc107;
    }
    .success-box {
        background-color: #e8f5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 0.3rem solid #4caf50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .small-font {
        font-size: 0.8rem;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    div[data-testid="stVerticalBlock"] div[style*="flex-direction: column;"] div[data-testid="stVerticalBlock"] {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .custom-tab {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .positive {
        color: #4caf50;
        font-weight: 600;
    }
    .negative {
        color: #f44336;
        font-weight: 600;
    }
    .neutral {
        color: #2196f3;
        font-weight: 600;
    }
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 1rem;
    }
    .card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        flex: 1;
        min-width: 250px;
    }
    div.stButton > button {
        background-color: #4e8df5;
        color: white;
        border-radius: 0.3rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #3c7ae4;
    }
    .explanation {
        background-color: #f9f9f9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 0.5rem;
        border-left: 3px solid #9e9e9e;
        font-size: 0.9rem;
    }
    .glossary-term {
        cursor: help;
        text-decoration: underline dotted;
        color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# Función para mostrar un spinner mientras carga
@st.cache_data
def load_data_with_spinner():
    with st.spinner('Cargando datos del World Values Survey...'):
        time.sleep(0.5)  # Simula tiempo de carga para mejor UX
        return load_data()

# Función para cargar y preparar los datos
@st.cache_data
def load_data():
    # Cargar datos del Excel real
    MISSING = [-1, -2, -4, -5]  # Códigos para valores faltantes
    
    # Cargar el Excel
    df = pd.read_excel("WVS_subset_felicidad.xlsx", engine='openpyxl')
    
    # Reemplazar valores faltantes
    df.replace(MISSING, np.nan, inplace=True)
    
    # Recodificaciones siguiendo el código original
    df['happiness'] = 5 - df['Q46']  # Invertir felicidad
    
    # Renombrar columnas
    df.rename(columns={
        'Q262': 'age', 'Q50': 'fin_sat', 'Q49': 'life_sat',
        'Q164': 'god_imp'
    }, inplace=True)
    
    # Invertir escalas 1-4
    df['imp_family'] = df['Q1'].apply(lambda x: 5-x if pd.notnull(x) else x)
    df['imp_friends'] = df['Q2'].apply(lambda x: 5-x if pd.notnull(x) else x)
    df['sec_rec'] = df['Q131'].apply(lambda x: 5-x if pd.notnull(x) else x)
    df['health'] = df['Q47'].apply(lambda x: 6-x if pd.notnull(x) else x)
    df['control'] = df['Q110'].apply(lambda x: 11-x if pd.notnull(x) else x)
    
    # Binarias/categóricas
    df['female'] = (df['Q260']==2).astype(float)
    df['soc_class'] = df['Q287'].astype(float)
    df['victim'] = (df['Q144']==1).astype(float)
    
    # Índice compuesto de crimen (invertir escalas y crear variables individuales)
    df['robos'] = df['Q132'].apply(lambda x: 5-x if pd.notnull(x) else x)
    df['alcohol'] = df['Q133'].apply(lambda x: 5-x if pd.notnull(x) else x)
    df['abuso_policial'] = df['Q134'].apply(lambda x: 5-x if pd.notnull(x) else x)
    df['racismo'] = df['Q135'].apply(lambda x: 5-x if pd.notnull(x) else x)
    df['drogas'] = df['Q136'].apply(lambda x: 5-x if pd.notnull(x) else x)
    df['violencia'] = df['Q137'].apply(lambda x: 5-x if pd.notnull(x) else x)
    df['acoso_sexual'] = df['Q138'].apply(lambda x: 5-x if pd.notnull(x) else x)
    
    CRIME_COLS = ['robos','alcohol','abuso_policial','racismo','drogas','violencia','acoso_sexual']
    df['indice_criminalidad'] = df[CRIME_COLS].mean(axis=1)
    
    # Índice compuesto de importancia de relaciones
    df['rel_import'] = df[['imp_family', 'imp_friends']].mean(axis=1)
    
    # Variables para el análisis
    cont_vars = ['happiness', 'age', 'fin_sat', 'health', 'life_sat',
                'god_imp', 'control', 'imp_family', 'imp_friends', 'rel_import', 'sec_rec', 'indice_criminalidad']
    cat_vars = ['female', 'soc_class', 'victim']
    
    # Listas para el análisis
    model_vars = ['age', 'female', 'soc_class', 'fin_sat', 'health', 'life_sat',
                 'god_imp', 'control', 'rel_import', 'sec_rec', 'indice_criminalidad', 'victim']
    
    # Crear un diccionario para las etiquetas amigables de las variables
    var_labels = {
        'age': 'Edad',
        'female': 'Género (Mujer)',
        'soc_class': 'Clase Social',
        'fin_sat': 'Satisfacción Financiera',
        'health': 'Salud',
        'life_sat': 'Satisfacción con la Vida',
        'god_imp': 'Importancia de Dios',
        'control': 'Control sobre la Vida',
        'rel_import': 'Importancia de Relaciones',
        'sec_rec': 'Seguridad Residencial',
        'indice_criminalidad': 'Índice de Criminalidad',
        'victim': 'Víctima de Delito',
        'happiness': 'Felicidad',
        'imp_family': 'Importancia de la Familia',
        'imp_friends': 'Importancia de los Amigos',
        'robos': 'Frecuencia de Robos',
        'alcohol': 'Problemas con Alcohol',
        'abuso_policial': 'Abuso Policial',
        'racismo': 'Racismo',
        'drogas': 'Problemas con Drogas',
        'violencia': 'Violencia',
        'acoso_sexual': 'Acoso Sexual'
    }
    
    # Crear un diccionario con descripciones de las variables
    var_descriptions = {
        'age': 'Edad del encuestado en años.',
        'female': 'Género del encuestado (0 = Hombre, 1 = Mujer).',
        'soc_class': 'Clase social auto-reportada (1 = Alta, 5 = Baja).',
        'fin_sat': 'Satisfacción con la situación financiera personal (1-10, mayor = más satisfecho).',
        'health': 'Estado de salud auto-reportado (1 = Muy bueno, 5 = Muy malo), recodificado para que mayor = mejor salud.',
        'life_sat': 'Satisfacción general con la vida (1-10, mayor = más satisfecho).',
        'god_imp': 'Importancia de Dios en la vida (1-10, mayor = más importante).',
        'control': 'Percepción de control sobre la propia vida (1-10, mayor = más control), recodificado.',
        'rel_import': 'Índice de importancia de las relaciones personales (1-4, mayor = más importante).',
        'sec_rec': 'Percepción de seguridad en el área residencial (1-4, mayor = más seguridad).',
        'indice_criminalidad': 'Índice compuesto sobre percepción de problemas de criminalidad en el área (1-4, mayor = más problemas).',
        'victim': 'Si ha sido víctima de algún delito en el último año (0 = No, 1 = Sí).',
        'happiness': 'Nivel de felicidad auto-reportado (1 = Muy infeliz, 4 = Muy feliz).'
    }
    
    # Crear un diccionario con las unidades o escalas de las variables
    var_units = {
        'age': 'años',
        'female': 'binaria (0-1)',
        'soc_class': 'ordinal (1-5)',
        'fin_sat': 'escala (1-10)',
        'health': 'ordinal (1-5)',
        'life_sat': 'escala (1-10)',
        'god_imp': 'escala (1-10)',
        'control': 'escala (1-10)',
        'rel_import': 'escala (1-4)',
        'sec_rec': 'ordinal (1-4)',
        'indice_criminalidad': 'escala (1-4)',
        'victim': 'binaria (0-1)',
        'happiness': 'ordinal (1-4)'
    }
    
    # Etiquetas para las categorías
    happiness_labels = ['Muy Infeliz', 'Infeliz', 'Feliz', 'Muy Feliz']
    class_labels = ['Alta', 'Media-Alta', 'Media', 'Media-Baja', 'Baja']
    gender_labels = ['Hombre', 'Mujer']
    victim_labels = ['No', 'Sí']
    
    # Dentro de la función load_data()
    # Países con su respectivo número de encuestados (si existe la columna country)
    if 'country' in df.columns:
        countries_count = df.groupby('country').size().sort_values(ascending=False)
    else:
        # Si no existe la columna, crear un Series vacío
        countries_count = pd.Series(dtype=int)
    return df, cont_vars, cat_vars, model_vars, var_labels, var_descriptions, var_units, happiness_labels, class_labels, gender_labels, victim_labels, countries_count, CRIME_COLS

# Función para calcular correlaciones con la felicidad
def calculate_correlations(df, vars_list, var_labels):
    corr_results = []
    for var in vars_list:
        if var != 'happiness':
            # Crear un DataFrame temporal con ambas variables
            # y eliminar filas donde cualquiera de las dos tenga valores faltantes
            temp_df = df[[var, 'happiness']].dropna()
            
            # Calcular correlación solo si hay suficientes datos
            if len(temp_df) > 5:  # Asegurar que hay suficientes datos para correlación
                rho, p = spearmanr(temp_df[var], temp_df['happiness'], nan_policy='omit')
                corr_results.append({
                    'Variable': var,
                    'Etiqueta': var_labels.get(var, var),
                    'Correlación (ρ)': rho,
                    'p-valor': p,
                    'Significativo': p < 0.05
                })
            else:
                corr_results.append({
                    'Variable': var,
                    'Etiqueta': var_labels.get(var, var),
                    'Correlación (ρ)': np.nan,
                    'p-valor': np.nan,
                    'Significativo': False
                })
    return pd.DataFrame(corr_results)

# Función para ajustar el modelo ordinal
def fit_ordinal_model(df, predictors):
    # Eliminar filas con valores faltantes
    model_df = df[predictors + ['happiness']].dropna()
    
    X = model_df[predictors]
    y = model_df['happiness']
    
    mod = OrderedModel(y, X, distr='logit')
    res = mod.fit(method='bfgs', disp=False)
    
    # Extraer coeficientes y p-valores
    results_df = pd.DataFrame({
        'Variable': res.params.index,
        'Coeficiente': res.params.values,
        'Error Estándar': res.bse.values,
        'p-valor': res.pvalues.values,
        'Significativo': res.pvalues < 0.05
    })

    # Filtrar variables no deseadas
    variables_a_ignorar = ['1.0/2.0', '2.0/3.0', '3.0/4.0']
    results_df = results_df[~results_df['Variable'].isin(variables_a_ignorar)]

    return res, results_df

# Función para calcular odd-ratios a partir de los coeficientes de regresión logística
def calculate_odds_ratios(model_results):
    odds_ratios = pd.DataFrame({
        'Variable': model_results['Variable'],
        'Odds Ratio': np.exp(model_results['Coeficiente']),
        'OR Lower CI': np.exp(model_results['Coeficiente'] - 1.96 * model_results['Error Estándar']),
        'OR Upper CI': np.exp(model_results['Coeficiente'] + 1.96 * model_results['Error Estándar']),
        'p-valor': model_results['p-valor'],
        'Significativo': model_results['Significativo']
    })
    return odds_ratios

# Función para generar predicciones marginales
def generate_marginal_predictions(model, result, base_values, var_to_vary, range_min, range_max, num_points=5):
    # Crear valores base
    base = pd.DataFrame([base_values])
    
    # Generar secuencia de valores para la variable a variar
    var_values = np.linspace(range_min, range_max, num_points)
    
    # Generar predicciones
    preds = []
    for val in var_values:
        # Copiar valores base y modificar la variable a variar
        temp_base = base.copy()
        temp_base[var_to_vary] = val
        
        # Obtener predicciones
        pred = model.predict(result.params, temp_base)
        preds.append(pred[0])
    
    # Convertir a array
    preds = np.array(preds)
    
    return var_values, preds

# Función para generar perfil de persona más feliz basado en los coeficientes del modelo
def generate_happy_profile(model_results, df, model_vars, var_labels):
    # Ordenar variables por coeficiente (de mayor a menor)
    sorted_vars = model_results.sort_values('Coeficiente', ascending=False)
    
    positive_vars = sorted_vars[sorted_vars['Coeficiente'] > 0]['Variable'].tolist()
    negative_vars = sorted_vars[sorted_vars['Coeficiente'] < 0]['Variable'].tolist()
    
    profile = {}
    
    # Para variables con efecto positivo, elegir valores altos
    for var in positive_vars:
        if var in ['female', 'victim']:  # Variables binarias
            if model_results.loc[model_results['Variable'] == var, 'Coeficiente'].values[0] > 0:
                profile[var] = 1
            else:
                profile[var] = 0
        else:  # Variables continuas
            # Usar el percentil 90 para variables positivas
            profile[var] = df[var].dropna().quantile(0.9)
            
    # Para variables con efecto negativo, elegir valores bajos
    for var in negative_vars:
        if var in ['female', 'victim']:  # Variables binarias
            if model_results.loc[model_results['Variable'] == var, 'Coeficiente'].values[0] < 0:
                profile[var] = 0
            else:
                profile[var] = 1
        else:  # Variables continuas
            # Usar el percentil 10 para variables negativas
            profile[var] = df[var].dropna().quantile(0.1)
    
    # Formatear el perfil para mostrar
    profile_display = []
    for var, value in profile.items():
        var_name = var_labels.get(var, var)
        if var in ['female', 'victim']:
            if var == 'female':
                value_text = 'Mujer' if value == 1 else 'Hombre'
            else:
                value_text = 'Sí' if value == 1 else 'No'
        elif var == 'soc_class':
            class_labels = ['Alta', 'Media-Alta', 'Media', 'Media-Baja', 'Baja']
            value_text = class_labels[int(value)-1] if 1 <= value <= 5 else f"{value:.1f}"
        else:
            value_text = f"{value:.1f}"
        
        profile_display.append({"Variable": var_name, "Valor Ideal": value_text})
    
    return pd.DataFrame(profile_display)

# Función para crear tarjetas de información en la pestaña de insights
def create_insight_card(title, content, icon="ℹ️"):
    st.markdown(f"""
    <div class="card">
    <h3>{icon} {title}</h3>
    <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

# Cargar datos
df, cont_vars, cat_vars, model_vars, var_labels, var_descriptions, var_units, happiness_labels, class_labels, gender_labels, victim_labels, countries_count, CRIME_COLS = load_data_with_spinner()

# TÍTULO DEL DASHBOARD
st.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #3a7bd5, #00d2ff); border-radius: 1rem; margin-bottom: 2rem;">
    <h1 style="color: white; margin-bottom: 0.5rem;">🌟 En Busca de la Felicidad</h1>
    <p style="color: white; font-size: 1.2rem; margin-bottom: 0.5rem;">Análisis interactivo de los determinantes de la felicidad humana</p>
    <p style="color: white; font-size: 0.9rem;">Basado en datos del World Values Survey - {0:,} personas.</p>
</div>
""".format(len(df)), unsafe_allow_html=True)

# Información sobre el dashboard
with st.expander("ℹ️ Acerca de este dashboard", expanded=False):
    st.markdown("""
    <div class="info-box">
    <h4>🔍 Acerca del estudio</h4>
    <p>Este dashboard analiza los factores que influyen en la felicidad auto-reportada utilizando datos del <b>World Values Survey</b>, 
    uno de los estudios más completos sobre valores humanos a nivel global. La escala de felicidad va de 1 (Muy infeliz) a 4 (Muy feliz).</p>
    
    <h4>📊 Análisis realizados</h4>
    <ul>
        <li><b>Análisis exploratorio:</b> Distribución de variables y estadísticas descriptivas</li>
        <li><b>Análisis bivariado:</b> Relaciones entre cada variable y la felicidad</li>
        <li><b>Modelo multivariado:</b> Regresión logística ordinal para determinar el efecto conjunto de factores</li>
        <li><b>Simulador:</b> Predicción interactiva del nivel de felicidad</li>
        <li><b>Análisis de interacciones:</b> Cómo interactúan diferentes variables entre sí</li>
        <li><b>Insights clave:</b> Hallazgos principales y recomendaciones</li>
    </ul>
    
    <h4>🧠 Modelo estadístico</h4>
    <p>Se utiliza una <span class="glossary-term" title="Modelo que tiene en cuenta el orden de las categorías de la variable dependiente.">regresión logística ordinal</span> para modelar cómo las diferentes variables 
    afectan la probabilidad de estar en cada nivel de felicidad, respetando el orden natural de los niveles.</p>
    
    <h4>🌍 Datos</h4>
    <p>Los datos provienen de la encuesta World Values Survey, que recoge información de personas de diversos países y contextos socioculturales. 
    La base de datos utilizada contiene {0:,} encuestados de {1} países.</p>
    </div>
    """.format(len(df), len(countries_count)), unsafe_allow_html=True)

# TABS PARA ORGANIZAR EL CONTENIDO
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 **Exploratorio**", 
    "🔄 **Bivariado**", 
    "📊 **Modelo**", 
    "🔮 **Simulador**", 
    "📉 **Interacciones**",
    "💡 **Insights**"
])

# TAB 1: ANÁLISIS EXPLORATORIO
with tab1:
    colored_header(
        label="Exploración de Variables",
        description="Analiza la distribución y características básicas de las variables",
        color_name="blue-70"
    )
    
    # Añadir mapa del mundo con distribución de felicidad por país si tenemos datos por país
    if 'country' in df.columns and len(countries_count) > 0:
        st.subheader("🌎 Distribución Global de la Felicidad")
        
        # Calcular la felicidad media por país
        happiness_by_country = df.groupby('country')['happiness'].mean().reset_index()
        happiness_by_country.columns = ['country', 'happiness_mean']
        
        # Contar el número de encuestados por país
        respondents_by_country = df.groupby('country').size().reset_index()
        respondents_by_country.columns = ['country', 'respondents']
        
        # Unir ambos datasets
        country_data = pd.merge(happiness_by_country, respondents_by_country, on='country')
        
        # Crear un mapa coroplético
        fig = px.choropleth(
            country_data,
            locations="country",
            locationmode="country names",
            color="happiness_mean",
            hover_name="country",
            hover_data={"happiness_mean": ":.2f", "respondents": True},
            color_continuous_scale=px.colors.sequential.YlGnBu,
            labels={"happiness_mean": "Felicidad Media", "respondents": "Encuestados"},
            title="Nivel Medio de Felicidad por País"
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            ),
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Ver datos de felicidad por país"):
            # Ordenar por felicidad media
            country_data_sorted = country_data.sort_values('happiness_mean', ascending=False)
            
            # Mostrar tabla con estilo
            st.dataframe(
                country_data_sorted,
                column_config={
                    "country": "País",
                    "happiness_mean": st.column_config.NumberColumn(
                        "Felicidad Media",
                        format="%.2f",
                        help="Nivel medio de felicidad (1-4)"
                    ),
                    "respondents": st.column_config.NumberColumn(
                        "Encuestados",
                        format="%d",
                        help="Número de encuestados"
                    )
                },
                use_container_width=True
            )
    
    # Distribución de la felicidad
    st.subheader("😊 Distribución de la Felicidad")
    
    # Layout con dos columnas para el gráfico y las estadísticas
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Gráfico de barras mejorado para felicidad
        # Gráfico de barras mejorado para felicidad
        happiness_counts = df['happiness'].value_counts().sort_index()
        happiness_pct = (happiness_counts / happiness_counts.sum() * 100).round(1)

        # Crear un DataFrame para el gráfico
        happiness_df = pd.DataFrame({
            'Nivel': [happiness_labels[int(i)-1] for i in happiness_counts.index],
            'Conteo': happiness_counts.values,
            'Porcentaje': happiness_pct.values
        })
        
        # Gráfico de barras con etiquetas de porcentaje
        fig_happiness = px.bar(
            happiness_df,
            x='Nivel',
            y='Conteo',
            text='Porcentaje',
            color='Nivel',
            labels={'Conteo': 'Número de Personas', 'Nivel': 'Nivel de Felicidad'},
            title='Distribución de los Niveles de Felicidad',
            color_discrete_map={
                'Muy Infeliz': '#e74c3c',
                'Infeliz': '#f39c12',
                'Feliz': '#2ecc71',
                'Muy Feliz': '#27ae60'
            }
        )
        
        fig_happiness.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        fig_happiness.update_layout(
            xaxis_title='Nivel de Felicidad',
            yaxis_title='Número de Personas',
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            height=400
        )
        
        st.plotly_chart(fig_happiness, use_container_width=True)
    
    with col2:
        # Métricas clave sobre la felicidad
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Nivel Medio de Felicidad",
            value=f"{df['happiness'].mean():.2f}",
            help="Escala de 1 (Muy infeliz) a 4 (Muy feliz)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        most_common = happiness_labels[int(df['happiness'].mode()[0])-1]
        st.metric(
            label="Nivel más Común",
            value=most_common,
            help="El nivel de felicidad reportado con mayor frecuencia"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        happy_pct = (df['happiness'] >= 3).mean() * 100
        st.metric(
            label="% Personas Felices",
            value=f"{happy_pct:.1f}%",
            help="Porcentaje de personas que reportan ser 'Felices' o 'Muy Felices'"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparación de felicidad entre hombres y mujeres
        if 'female' in df.columns:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            male_happiness = df[df['female'] == 0]['happiness'].mean()
            female_happiness = df[df['female'] == 1]['happiness'].mean()
            diff = female_happiness - male_happiness
            st.metric(
                label="Diferencia por Género",
                value=f"{diff:.2f}",
                delta=f"{'Mujeres' if diff > 0 else 'Hombres'} más felices",
                help="Diferencia en el nivel medio de felicidad entre mujeres y hombres"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Selector para variables continuas y categóricas
    st.subheader("🔍 Explorar Distribución de Variables")
    
    # Pestañas para variables continuas y categóricas
    explore_tabs = st.tabs(["Variables Continuas", "Variables Categóricas"])
    
    # Pestaña de variables continuas
    with explore_tabs[0]:
        # Seleccionar variable continua
        selected_cont_var = st.selectbox(
            "Selecciona una variable continua:", 
            [v for v in cont_vars if v != 'happiness'],
            format_func=lambda x: var_labels.get(x, x),
            key="explore_cont_var"
        )
        
        # Descripción de la variable
        if selected_cont_var in var_descriptions:
            st.markdown(f'<div class="explanation"><p><b>{var_labels.get(selected_cont_var, selected_cont_var)}:</b> {var_descriptions[selected_cont_var]}</p></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma con KDE para la variable seleccionada
            fig_cont = px.histogram(
                df.dropna(subset=[selected_cont_var]), 
                x=selected_cont_var,
                nbins=30,
                marginal="rug",
                opacity=0.8,
                color_discrete_sequence=['#2ecc71'],
                labels={selected_cont_var: var_labels.get(selected_cont_var, selected_cont_var)},
                title=f'Distribución de {var_labels.get(selected_cont_var, selected_cont_var)}'
            )
            
            # Añadir línea KDE
            hist_data = [df[selected_cont_var].dropna()]
            group_labels = [var_labels.get(selected_cont_var, selected_cont_var)]
            
            # Configurar el histograma
            fig_cont.update_layout(
                xaxis_title=var_labels.get(selected_cont_var, selected_cont_var),
                yaxis_title="Frecuencia",
                height=400
            )
            
            st.plotly_chart(fig_cont, use_container_width=True)
        
        with col2:
            # Boxplot para la variable seleccionada
            fig_box = px.box(
                df.dropna(subset=[selected_cont_var]),
                y=selected_cont_var,
                points="all",
                notched=True,
                labels={selected_cont_var: var_labels.get(selected_cont_var, selected_cont_var)},
                title=f'Boxplot de {var_labels.get(selected_cont_var, selected_cont_var)}'
            )
            
            fig_box.update_traces(
                marker=dict(
                    color='#3498db',
                    opacity=0.7,
                    size=3
                ),
                line=dict(color='#1a5276')
            )
            
            fig_box.update_layout(
                yaxis_title=var_labels.get(selected_cont_var, selected_cont_var),
                height=400
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Mostrar estadísticas descriptivas para la variable seleccionada
        stats_df = df[selected_cont_var].describe().round(2).to_frame().T
        stats_df.index = [var_labels.get(selected_cont_var, selected_cont_var)]
        
        st.markdown("### 📊 Estadísticas Descriptivas")
        
        # Crear una tabla estilizada para las estadísticas
        stats_cols = st.columns(8)
        metrics = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        metric_names = ["N", "Media", "Desv. Est.", "Mínimo", "Q1", "Mediana", "Q3", "Máximo"]
        
        for i, (col, metric, name) in enumerate(zip(stats_cols, metrics, metric_names)):
            with col:
                st.markdown(f"<p style='text-align: center; font-weight: bold;'>{name}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>{stats_df[metric].values[0]:,}</p>", unsafe_allow_html=True)

    # Pestaña de variables categóricas
    with explore_tabs[1]:
        # Seleccionar variable categórica
        selected_cat_var = st.selectbox(
            "Selecciona una variable categórica:", 
            cat_vars,
            format_func=lambda x: var_labels.get(x, x),
            key="explore_cat_var"
        )
        
        # Descripción de la variable
        if selected_cat_var in var_descriptions:
            st.markdown(f'<div class="explanation"><p><b>{var_labels.get(selected_cat_var, selected_cat_var)}:</b> {var_descriptions[selected_cat_var]}</p></div>', unsafe_allow_html=True)
            
        # Contar los valores de la variable categórica
        cat_counts = df[selected_cat_var].value_counts().sort_index()
        cat_pct = (cat_counts / cat_counts.sum() * 100).round(1)
        
        # Crear un DataFrame para el gráfico
        cat_df = pd.DataFrame({
            'Categoría': cat_counts.index,
            'Conteo': cat_counts.values,
            'Porcentaje': cat_pct.values
        })
        
        # Etiquetas personalizadas para las categorías
        if selected_cat_var == 'female':
            cat_df['Etiqueta'] = [gender_labels[int(i)] for i in cat_df['Categoría']]
        elif selected_cat_var == 'victim':
            cat_df['Etiqueta'] = [victim_labels[int(i)] for i in cat_df['Categoría']]
        elif selected_cat_var == 'soc_class':
            cat_df['Etiqueta'] = [class_labels[int(i)-1] if i <= 5 else f'Otro ({i})' for i in cat_df['Categoría']]
        else:
            cat_df['Etiqueta'] = cat_df['Categoría'].astype(str)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Gráfico de barras con etiquetas de porcentaje
            fig_cat = px.bar(
                cat_df,
                x='Etiqueta',
                y='Conteo',
                text='Porcentaje',
                color='Etiqueta',
                labels={'Conteo': 'Número de Personas', 'Etiqueta': var_labels.get(selected_cat_var, selected_cat_var)},
                title=f'Distribución de {var_labels.get(selected_cat_var, selected_cat_var)}'
            )
            
            fig_cat.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            
            fig_cat.update_layout(
                xaxis_title=var_labels.get(selected_cat_var, selected_cat_var),
                yaxis_title='Número de Personas',
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            # Gráfico de pastel para la variable categórica
            fig_pie = px.pie(
                cat_df,
                values='Conteo',
                names='Etiqueta',
                title=f'Proporción de {var_labels.get(selected_cat_var, selected_cat_var)}',
                hover_data=['Porcentaje'],
                labels={'Etiqueta': var_labels.get(selected_cat_var, selected_cat_var)}
            )
            
            fig_pie.update_traces(
                textinfo='percent+label',
                pull=[0.05 if i == cat_df['Conteo'].idxmax() else 0 for i in range(len(cat_df))]
            )
            
            fig_pie.update_layout(
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Mostrar estadísticas descriptivas para todas las variables
    with st.expander("📋 Ver Estadísticas Descriptivas Completas", expanded=False):
        st.subheader("Estadísticas Descriptivas")
        
        # Calcular estadísticas descriptivas
        desc_df = df[cont_vars].describe().round(2).T
        desc_df = desc_df.reset_index()
        desc_df.columns = ['Variable'] + list(desc_df.columns[1:])
        
        # Añadir etiquetas de variables
        desc_df['Etiqueta'] = desc_df['Variable'].map(lambda x: var_labels.get(x, x))
        
        # Reorganizar las columnas
        desc_df = desc_df[['Variable', 'Etiqueta', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        
        # Mostrar tabla estilizada
        st.dataframe(
            desc_df,
            column_config={
                "Variable": "Código",
                "Etiqueta": "Variable",
                "count": st.column_config.NumberColumn("N", format="%d"),
                "mean": st.column_config.NumberColumn("Media", format="%.2f"),
                "std": st.column_config.NumberColumn("Desv. Est.", format="%.2f"),
                "min": st.column_config.NumberColumn("Mín", format="%.2f"),
                "25%": st.column_config.NumberColumn("Q1", format="%.2f"),
                "50%": st.column_config.NumberColumn("Mediana", format="%.2f"),
                "75%": st.column_config.NumberColumn("Q3", format="%.2f"),
                "max": st.column_config.NumberColumn("Máx", format="%.2f")
            },
            use_container_width=True
        )
        
        # Mostrar info de variables categóricas
        st.subheader("Variables Categóricas")
        
        for var in cat_vars:
            counts = df[var].value_counts().sort_index()
            percent = counts / counts.sum() * 100
            
            cat_stats = pd.DataFrame({
                'Categoría': counts.index,
                'Conteo': counts.values,
                'Porcentaje': percent.values.round(1)
            })
            
            # Convertir categorías numéricas a etiquetas
            if var == 'female':
                cat_stats['Etiqueta'] = [gender_labels[int(i)] for i in cat_stats['Categoría']]
            elif var == 'victim':
                cat_stats['Etiqueta'] = [victim_labels[int(i)] for i in cat_stats['Categoría']]
            elif var == 'soc_class':
                cat_stats['Etiqueta'] = [class_labels[int(i)-1] if i <= 5 else f'Otro ({i})' for i in cat_stats['Categoría']]
            else:
                cat_stats['Etiqueta'] = cat_stats['Categoría'].astype(str)
            
            st.markdown(f"#### {var_labels.get(var, var)}")
            
            # Mostrar tabla de frecuencias
            st.dataframe(
                cat_stats,
                column_config={
                    "Categoría": "Valor",
                    "Etiqueta": "Etiqueta",
                    "Conteo": st.column_config.NumberColumn("Conteo", format="%d"),
                    "Porcentaje": st.column_config.NumberColumn("Porcentaje", format="%.1f%%")
                },
                use_container_width=True
            )
    
    # Gráfico de matriz de correlación
    with st.expander("🔄 Ver Matriz de Correlaciones"):
        # Calcular matriz de correlación
        correlation_vars = [v for v in cont_vars if v != 'happiness'] + ['happiness']
        corr_matrix = df[correlation_vars].corr(method='spearman').round(2)
        
        # Crear un mapa de calor de correlaciones
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(x="Variable", y="Variable", color="Correlación"),
            x=[var_labels.get(v, v) for v in correlation_vars],
            y=[var_labels.get(v, v) for v in correlation_vars],
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        
        fig_corr.update_layout(
            title='Matriz de Correlación de Spearman',
            height=600,
            xaxis={'side': 'bottom'}
        )
        
        # Añadir los valores de correlación como texto
        for i, var1 in enumerate(correlation_vars):
            for j, var2 in enumerate(correlation_vars):
                fig_corr.add_annotation(
                    x=var_labels.get(var2, var2),
                    y=var_labels.get(var1, var1),
                    text=str(corr_matrix.iloc[i, j]),
                    showarrow=False,
                    font=dict(color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white', size=10)
                )
        
        st.plotly_chart(fig_corr, use_container_width=True)

# TAB 2: ANÁLISIS BIVARIADO
with tab2:
    colored_header(
        label="Relaciones Bivariadas con la Felicidad",
        description="Explora cómo cada variable se relaciona con los niveles de felicidad",
        color_name="blue-70"
    )
    
    # Correlaciones con la felicidad
    st.subheader("📊 Panorama General de Relaciones")
    
    # Calcular todas las correlaciones con felicidad
    corr_df = calculate_correlations(df, cont_vars + cat_vars, var_labels)
    
    # Ordenar por magnitud absoluta de correlación (solo para filas con datos)
    valid_corr = corr_df.dropna(subset=['Correlación (ρ)'])
    
    if len(valid_corr) > 0:
        valid_corr['Magnitud'] = valid_corr['Correlación (ρ)'].abs()
        valid_corr = valid_corr.sort_values('Magnitud', ascending=False)
        
        # Gráfico de barras horizontales para las correlaciones
        fig_corr_bars = px.bar(
            valid_corr.head(10),  # Top 10 correlaciones
            y='Etiqueta',
            x='Correlación (ρ)',
            color='Correlación (ρ)',
            color_continuous_scale='RdBu_r',
            labels={'Correlación (ρ)': 'Correlación con Felicidad', 'Etiqueta': 'Variable'},
            title='Top 10 Variables por Correlación con Felicidad'
        )
        
        fig_corr_bars.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=400
        )
        
        st.plotly_chart(fig_corr_bars, use_container_width=True)
        
        # Resumen de las principales correlaciones en texto
        top_pos = valid_corr[valid_corr['Correlación (ρ)'] > 0].head(3)
        top_neg = valid_corr[valid_corr['Correlación (ρ)'] < 0].head(3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Variables con Correlación Positiva más Fuerte")
            for _, row in top_pos.iterrows():
                st.markdown(f"- **{row['Etiqueta']}**: ρ = {row['Correlación (ρ)']:.3f} ({'+' if row['Correlación (ρ)'] > 0 else ''}{row['Correlación (ρ)']:.3f})")
        
        with col2:
            st.markdown("#### Variables con Correlación Negativa más Fuerte")
            for _, row in top_neg.iterrows():
                st.markdown(f"- **{row['Etiqueta']}**: ρ = {row['Correlación (ρ)']:.3f} ({'+' if row['Correlación (ρ)'] > 0 else ''}{row['Correlación (ρ)']:.3f})")
        
        with st.expander("📋 Ver Tabla Completa de Correlaciones"):
            # Formatear tabla para mostrar
            display_corr = valid_corr.drop('Magnitud', axis=1).copy()
            
            # Estilizar la tabla de correlaciones
            st.dataframe(
                display_corr,
                column_config={
                    "Variable": st.column_config.TextColumn("Código"),
                    "Etiqueta": st.column_config.TextColumn("Variable"),
                    "Correlación (ρ)": st.column_config.NumberColumn(
                        "Correlación de Spearman (ρ)", 
                        format="%.3f"
                    ),
                    "p-valor": st.column_config.NumberColumn(
                        "p-valor",
                        format="%.4f"
                    ),
                    "Significativo": st.column_config.CheckboxColumn(
                        "Significativo (p<0.05)"
                    )
                },
                use_container_width=True
            )
    else:
        st.warning("No hay suficientes datos para calcular correlaciones.")
    
    # Análisis bivariado detallado
    st.markdown("---")
    st.subheader("🔍 Análisis Detallado por Variable")
    
    # Seleccionar tipo de variable para analizar
    var_type = st.radio(
        "Tipo de variable a analizar:",
        ["Continua", "Categórica"],
        key="bivariate_var_type",
        horizontal=True
    )
    
    if var_type == "Continua":
        # Seleccionar variable continua
        selected_var = st.selectbox(
            "Selecciona una variable continua:",
            [v for v in cont_vars if v != 'happiness'],
            format_func=lambda x: var_labels.get(x, x),
            key="bivariate_cont_var"
        )
        
        # Mostrar descripción de la variable
        if selected_var in var_descriptions:
            st.markdown(f'<div class="explanation"><p><b>{var_labels.get(selected_var, selected_var)}:</b> {var_descriptions[selected_var]}</p></div>', unsafe_allow_html=True)
        
        # Filtrar filas con valores faltantes
        scatter_df = df[[selected_var, 'happiness']].dropna()
        
        if len(scatter_df) > 5:  # Asegurarse de que hay suficientes datos
            # Layout de dos columnas
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de dispersión con jitter y tendencia
                jitter = np.random.normal(0, 0.05, size=len(scatter_df))
                scatter_df['happiness_jitter'] = scatter_df['happiness'] + jitter
                
                fig_scatter = px.scatter(
                    scatter_df,
                    x=selected_var,
                    y='happiness_jitter',
                    color='happiness',
                    color_discrete_map={
                        1: '#e74c3c',
                        2: '#f39c12',
                        3: '#2ecc71',
                        4: '#27ae60'
                    },
                    labels={
                        selected_var: var_labels.get(selected_var, selected_var),
                        'happiness_jitter': 'Nivel de Felicidad',
                        'happiness': 'Nivel de Felicidad'
                    },
                    title=f'Relación entre {var_labels.get(selected_var, selected_var)} y Felicidad',
                    category_orders={'happiness': [1, 2, 3, 4]},
                    opacity=0.6
                )
                
                # Añadir línea de tendencia
                z = np.polyfit(scatter_df[selected_var], scatter_df['happiness'], 1)
                p = np.poly1d(z)
                
                x_range = np.linspace(scatter_df[selected_var].min(), scatter_df[selected_var].max(), 100)
                fig_scatter.add_trace(go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    name='Tendencia',
                    line=dict(color='red', width=2)
                ))
                
                fig_scatter.update_layout(
                    xaxis_title=var_labels.get(selected_var, selected_var),
                    yaxis_title='Nivel de Felicidad',
                    height=400,
                    legend_title_text='Nivel de Felicidad'
                )
                
                # Actualizar ejes Y para mostrar etiquetas de felicidad
                fig_scatter.update_yaxes(
                    tickvals=[1, 2, 3, 4],
                    ticktext=happiness_labels
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Boxplot por nivel de felicidad (más claro y atractivo)
                fig_box = px.box(
                    scatter_df,
                    x='happiness',
                    y=selected_var,
                    color='happiness',
                    category_orders={'happiness': [1, 2, 3, 4]},
                    labels={
                        'happiness': 'Nivel de Felicidad',
                        selected_var: var_labels.get(selected_var, selected_var)
                    },
                    title=f'Distribución de {var_labels.get(selected_var, selected_var)} por Nivel de Felicidad',
                    color_discrete_map={
                        1: '#e74c3c',
                        2: '#f39c12',
                        3: '#2ecc71',
                        4: '#27ae60'
                    }
                )
                
                fig_box.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=[1, 2, 3, 4],
                        ticktext=happiness_labels
                    ),
                    yaxis_title=var_labels.get(selected_var, selected_var),
                    height=400,
                    legend_title_text='Nivel de Felicidad'
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Estadísticas por grupo
            st.subheader("📊 Estadísticas por Nivel de Felicidad")
            
            # Calcular estadísticas para cada nivel de felicidad
            group_stats = scatter_df.groupby('happiness')[selected_var].agg(['mean', 'median', 'std', 'count']).reset_index()
            group_stats.columns = ['Nivel', 'Media', 'Mediana', 'Desv. Est.', 'N']
            
            # Añadir etiquetas de felicidad
            group_stats['Etiqueta'] = group_stats['Nivel'].apply(lambda x: happiness_labels[int(x)-1])
            
            # Reordenar columnas
            group_stats = group_stats[['Nivel', 'Etiqueta', 'Media', 'Mediana', 'Desv. Est.', 'N']]
            
            # Mostrar estadísticas en una tabla estilizada
            st.dataframe(
                group_stats,
                column_config={
                    "Nivel": "Nivel",
                    "Etiqueta": "Descripción",
                    "Media": st.column_config.NumberColumn("Media", format="%.2f"),
                    "Mediana": st.column_config.NumberColumn("Mediana", format="%.2f"),
                    "Desv. Est.": st.column_config.NumberColumn("Desv. Est.", format="%.2f"),
                    "N": st.column_config.NumberColumn("N", format="%d")
                },
                use_container_width=True
            )
            
            # Calcular correlación de Spearman
            rho, p = spearmanr(scatter_df[selected_var], scatter_df['happiness'])
            
            # Mostrar correlación en una tarjeta estilizada
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            
            corr_cols = st.columns([1, 1])
            
            with corr_cols[0]:
                st.markdown(f"### Correlación de Spearman")
                st.markdown(f"**ρ = {rho:.3f}** (p = {p:.4f})")
                st.markdown(f"**Interpretación:** Correlación <span class='{'positive' if rho > 0 else 'negative' if rho < 0 else 'neutral'}'>{'positiva' if rho > 0 else 'negativa' if rho < 0 else 'nula'}</span> {'y estadísticamente significativa' if p < 0.05 else 'pero no estadísticamente significativa'} (p {'<' if p < 0.05 else '>'} 0.05).", unsafe_allow_html=True)
            
            with corr_cols[1]:
                corr_strength = "fuerte" if abs(rho) > 0.5 else "moderada" if abs(rho) > 0.3 else "débil" if abs(rho) > 0.1 else "muy débil o inexistente"
                direction = "positiva" if rho > 0 else "negativa" if rho < 0 else "neutral"
                
                st.markdown(f"### Conclusión")
                st.markdown(f"Existe una relación {corr_strength} y {direction} entre **{var_labels.get(selected_var, selected_var)}** y **Felicidad**.")
                
                # Interpretar el significado
                if rho > 0:
                    st.markdown(f"A mayor {var_labels.get(selected_var, selected_var).lower()}, mayor tiende a ser el nivel de felicidad reportado.")
                elif rho < 0:
                    st.markdown(f"A mayor {var_labels.get(selected_var, selected_var).lower()}, menor tiende a ser el nivel de felicidad reportado.")
                else:
                    st.markdown(f"No parece existir una relación clara entre {var_labels.get(selected_var, selected_var).lower()} y felicidad.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.warning("No hay suficientes datos para realizar el análisis bivariado.")
    
    else:  # Categórica
        # Seleccionar variable categórica
        selected_var = st.selectbox(
            "Selecciona una variable categórica:",
            cat_vars,
            format_func=lambda x: var_labels.get(x, x),
            key="bivariate_cat_var"
        )
        
        # Mostrar descripción de la variable
        if selected_var in var_descriptions:
            st.markdown(f'<div class="explanation"><p><b>{var_labels.get(selected_var, selected_var)}:</b> {var_descriptions[selected_var]}</p></div>', unsafe_allow_html=True)
        
        # Filtrar filas con valores faltantes
        crosstab_df = df[[selected_var, 'happiness']].dropna()
        
        if len(crosstab_df) > 10:  # Asegurar que hay suficientes datos
            # Layout de dos columnas
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de barras apiladas (proporcional)
                # Crear tabla de contingencia
                crosstab = pd.crosstab(crosstab_df[selected_var], crosstab_df['happiness'], normalize='index') * 100
                crosstab = crosstab.reset_index()
                
                # Convertir a formato largo para Plotly
                crosstab_long = pd.melt(
                    crosstab,
                    id_vars=[selected_var],
                    var_name='happiness',
                    value_name='percentage'
                )
                
                # Añadir etiquetas para las categorías
                if selected_var == 'female':
                    crosstab_long['category_label'] = crosstab_long[selected_var].apply(lambda x: gender_labels[int(x)])
                elif selected_var == 'victim':
                    crosstab_long['category_label'] = crosstab_long[selected_var].apply(lambda x: victim_labels[int(x)])
                elif selected_var == 'soc_class':
                    crosstab_long['category_label'] = crosstab_long[selected_var].apply(lambda x: class_labels[int(x)-1] if x <= 5 else f'Otro ({x})')
                else:
                    crosstab_long['category_label'] = crosstab_long[selected_var].astype(str)
                
                # Añadir etiquetas para los niveles de felicidad
                crosstab_long['happiness_label'] = crosstab_long['happiness'].apply(lambda x: happiness_labels[int(x)-1])
                
                # Gráfico de barras apiladas mejorado
                fig_stacked = px.bar(
                    crosstab_long,
                    x='category_label',
                    y='percentage',
                    color='happiness_label',
                    barmode='stack',
                    labels={
                        'percentage': 'Porcentaje (%)',
                        'category_label': var_labels.get(selected_var, selected_var),
                        'happiness_label': 'Nivel de Felicidad'
                    },
                    title=f'Proporción de Felicidad por {var_labels.get(selected_var, selected_var)}',
                    color_discrete_map={
                        'Muy Infeliz': '#e74c3c',
                        'Infeliz': '#f39c12',
                        'Feliz': '#2ecc71',
                        'Muy Feliz': '#27ae60'
                    }
                )
                
                # Mejorar el layout
                fig_stacked.update_layout(
                    xaxis_title=var_labels.get(selected_var, selected_var),
                    yaxis_title='Porcentaje (%)',
                    height=400
                )
                
                st.plotly_chart(fig_stacked, use_container_width=True)
                
                # Explicación del gráfico
                st.markdown('<div class="explanation">', unsafe_allow_html=True)
                st.markdown("**Interpretación:** Este gráfico muestra la distribución porcentual de los niveles de felicidad para cada categoría. Las barras apiladas permiten comparar cómo se distribuye la felicidad entre los diferentes grupos.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Gráfico de barras para la media de felicidad por categoría
                happiness_means = crosstab_df.groupby(selected_var)['happiness'].agg(['mean', 'sem', 'count']).reset_index()
                
                # Añadir etiquetas para las categorías
                if selected_var == 'female':
                    happiness_means['category_label'] = happiness_means[selected_var].apply(lambda x: gender_labels[int(x)])
                elif selected_var == 'victim':
                    happiness_means['category_label'] = happiness_means[selected_var].apply(lambda x: victim_labels[int(x)])
                elif selected_var == 'soc_class':
                    happiness_means['category_label'] = happiness_means[selected_var].apply(lambda x: class_labels[int(x)-1] if x <= 5 else f'Otro ({x})')
                else:
                    happiness_means['category_label'] = happiness_means[selected_var].astype(str)
                
                # Gráfico de barras con error
                fig_means = px.bar(
                    happiness_means,
                    x='category_label',
                    y='mean',
                    error_y='sem',
                    labels={
                        'mean': 'Felicidad Media (1-4)',
                        'category_label': var_labels.get(selected_var, selected_var)
                    },
                    title=f'Felicidad Media por {var_labels.get(selected_var, selected_var)}',
                    color='category_label'
                )
                
                # Añadir línea de referencia para la media global
                global_mean = df['happiness'].mean()
                
                fig_means.add_shape(
                    type="line",
                    line=dict(dash="dash", color="gray", width=2),
                    y0=global_mean,
                    y1=global_mean,
                    x0=-0.5,
                    x1=len(happiness_means) - 0.5
                )
                
                fig_means.add_annotation(
                    x=len(happiness_means) - 0.5,
                    y=global_mean,
                    text=f"Media Global: {global_mean:.2f}",
                    showarrow=False,
                    yshift=10,
                    xshift=-5,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                )
                
                # Mejorar el layout
                fig_means.update_layout(
                    xaxis_title=var_labels.get(selected_var, selected_var),
                    yaxis_title='Felicidad Media (1-4)',
                    height=400,
                    showlegend=False,
                    yaxis=dict(range=[1, 4])
                )
                
                st.plotly_chart(fig_means, use_container_width=True)
                
                # Explicación del gráfico
                st.markdown('<div class="explanation">', unsafe_allow_html=True)
                st.markdown("**Interpretación:** Este gráfico muestra el nivel medio de felicidad para cada categoría. Las barras de error representan el error estándar de la media. La línea punteada indica la media global de felicidad.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Tabla de contingencia y prueba estadística
            st.subheader("📊 Análisis Estadístico")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Tabla de contingencia con frecuencias absolutas
                st.markdown("#### Tabla de Contingencia")
                contingency_table = pd.crosstab(crosstab_df[selected_var], crosstab_df['happiness'])
                
                # Renombrar índices y columnas para mayor claridad
                if selected_var == 'female':
                    contingency_table.index = [gender_labels[int(i)] for i in contingency_table.index]
                elif selected_var == 'victim':
                    contingency_table.index = [victim_labels[int(i)] for i in contingency_table.index]
                elif selected_var == 'soc_class':
                    contingency_table.index = [class_labels[int(i)-1] if i <= 5 else f'Otro ({i})' for i in contingency_table.index]
                
                contingency_table.columns = happiness_labels
                
                # Añadir totales
                contingency_table['Total'] = contingency_table.sum(axis=1)
                contingency_table.loc['Total'] = contingency_table.sum()
                
                st.dataframe(contingency_table, use_container_width=True)
            
            with col4:
                # Prueba estadística y resultado
                st.markdown("#### Prueba Estadística")
                
                if crosstab_df[selected_var].nunique() == 2:
                    # Para variables binarias usamos Mann-Whitney (equivalente)
                    g0 = crosstab_df[crosstab_df[selected_var]==0]['happiness']
                    g1 = crosstab_df[crosstab_df[selected_var]==1]['happiness']
                    
                    # Solo calcular si ambos grupos tienen datos
                    if len(g0) > 0 and len(g1) > 0:
                        U, p_val = mannwhitneyu(g0, g1)
                        test_name = "Mann-Whitney U"
                        test_stat = U
                        
                        # Mostrar resultados
                        st.markdown(f"**Prueba:** {test_name}")
                        st.markdown(f"**Estadístico:** U = {test_stat:.1f}")
                        st.markdown(f"**p-valor:** {p_val:.4f}")
                        
                        # Interpretación
                        if p_val < 0.05:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown("**Conclusión:** Existe una diferencia estadísticamente significativa en los niveles de felicidad entre los grupos (p < 0.05).")
                            
                            # Determinar qué grupo tiene mayor felicidad
                            group0_mean = g0.mean()
                            group1_mean = g1.mean()
                            
                            if group0_mean != group1_mean:
                                higher_group = gender_labels[0] if selected_var == 'female' and group0_mean > group1_mean else gender_labels[1] if selected_var == 'female' else victim_labels[0] if selected_var == 'victim' and group0_mean > group1_mean else victim_labels[1]
                                lower_group = gender_labels[1] if selected_var == 'female' and group0_mean > group1_mean else gender_labels[0] if selected_var == 'female' else victim_labels[1] if selected_var == 'victim' and group0_mean > group1_mean else victim_labels[0]
                                
                                st.markdown(f"El grupo **{higher_group}** muestra niveles de felicidad significativamente mayores que el grupo **{lower_group}**.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown("**Conclusión:** No hay evidencia suficiente para afirmar que existe una diferencia en los niveles de felicidad entre los grupos (p > 0.05).")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No hay suficientes datos en alguno de los grupos para realizar la prueba.")
                else:
                    # Para variables con más categorías usamos Kruskal-Wallis
                    # Agrupar por categorías
                    groups = []
                    group_labels = []
                    
                    for val in crosstab_df[selected_var].unique():
                        group = crosstab_df[crosstab_df[selected_var]==val]['happiness'].values
                        if len(group) > 0:
                            groups.append(group)
                            
                            # Generar etiqueta para el grupo
                            if selected_var == 'female':
                                label = gender_labels[int(val)]
                            elif selected_var == 'victim':
                                label = victim_labels[int(val)]
                            elif selected_var == 'soc_class':
                                label = class_labels[int(val)-1] if val <= 5 else f'Otro ({val})'
                            else:
                                label = str(val)
                            
                            group_labels.append(label)
                    
                    # Solo calcular si hay al menos dos grupos con datos
                    if len(groups) >= 2:
                        H, p_val = kruskal(*groups)
                        test_name = "Kruskal-Wallis H"
                        test_stat = H
                        
                        # Mostrar resultados
                        st.markdown(f"**Prueba:** {test_name}")
                        st.markdown(f"**Estadístico:** H = {test_stat:.1f}")
                        st.markdown(f"**p-valor:** {p_val:.4f}")
                        
                        # Interpretación
                        if p_val < 0.05:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown("**Conclusión:** Existen diferencias estadísticamente significativas en los niveles de felicidad entre los grupos (p < 0.05).")
                            
                            # Mostrar medias por grupo para ayudar en la interpretación
                            group_means = crosstab_df.groupby(selected_var)['happiness'].mean()
                            sorted_means = group_means.sort_values(ascending=False)
                            
                            st.markdown("**Ranking de grupos por nivel medio de felicidad:**")
                            
                            for i, (val, mean) in enumerate(sorted_means.items()):
                                # Generar etiqueta para el grupo
                                if selected_var == 'female':
                                    label = gender_labels[int(val)]
                                elif selected_var == 'victim':
                                    label = victim_labels[int(val)]
                                elif selected_var == 'soc_class':
                                    label = class_labels[int(val)-1] if val <= 5 else f'Otro ({val})'
                                else:
                                    label = str(val)
                                
                                st.markdown(f"{i+1}. **{label}**: {mean:.2f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown("**Conclusión:** No hay evidencia suficiente para afirmar que existen diferencias en los niveles de felicidad entre los grupos (p > 0.05).")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No hay suficientes grupos con datos para realizar la prueba.")
        else:
            st.warning("No hay suficientes datos para realizar el análisis bivariado.")

# TAB 3: RESULTADOS DEL MODELO
with tab3:
    colored_header(
        label="Modelo Multivariado",
        description="Análisis del modelo ordinal logístico para predecir felicidad",
        color_name="blue-70"
    )
    
    # Descripción del modelo
    st.markdown("""
    <div class="info-box">
    <h4>🧠 Modelo Estadístico</h4>
    <p>Se utiliza una <b>regresión logística ordinal</b> para modelar la probabilidad de cada nivel de felicidad en función de múltiples variables predictoras. 
    Este tipo de modelo es adecuado cuando la variable dependiente es ordinal (tiene un orden natural) como los niveles de felicidad.</p>
    
    <p>El modelo estima <b>coeficientes</b> para cada variable, donde:</p>
    <ul>
        <li>Coeficientes <span class="positive">positivos</span> indican que la variable está asociada con <b>mayor</b> probabilidad de niveles más altos de felicidad</li>
        <li>Coeficientes <span class="negative">negativos</span> indican que la variable está asociada con <b>menor</b> probabilidad de niveles más altos de felicidad</li>
        <li>La <b>magnitud</b> del coeficiente indica la fuerza de la asociación</li>
        <li>El <b>p-valor</b> indica si la asociación es estadísticamente significativa (p < 0.05)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar si hay suficientes datos para el modelo
    model_data = df[model_vars + ['happiness']].dropna()
    
    if len(model_data) > len(model_vars) + 10:  # Comprobar que hay suficientes observaciones
        # Ajustar el modelo con spinner
        with st.spinner('Ajustando modelo ordinal logístico...'):
            try:
                model, model_results = fit_ordinal_model(df, model_vars)
                
                # Calcular odd ratios para una interpretación más intuitiva
                odds_ratios = calculate_odds_ratios(model_results)
                
                # Contenedor para métricas clave del modelo
                st.subheader("🎯 Rendimiento del Modelo")
                
                metrics_cols = st.columns(4)
                
                with metrics_cols[0]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Observaciones",
                        value=f"{model.nobs:,}",
                        help="Número de observaciones utilizadas para ajustar el modelo"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metrics_cols[1]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Log-Likelihood",
                        value=f"{model.llf:.2f}",
                        help="Logaritmo de la verosimilitud del modelo; valores más altos indican mejor ajuste"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metrics_cols[2]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="AIC",
                        value=f"{model.aic:.2f}",
                        help="Criterio de Información de Akaike; valores más bajos indican mejor ajuste"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metrics_cols[3]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    sig_count = model_results['Significativo'].sum()
                    total_count = len(model_results)
                    st.metric(
                        label="Variables Significativas",
                        value=f"{sig_count}/{total_count}",
                        help="Número de variables que tienen un efecto estadísticamente significativo en la felicidad"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Visualización de coeficientes
                st.subheader("📊 Coeficientes del Modelo")
                
                # Pestañas para diferentes visualizaciones
                coef_tabs = st.tabs(["Gráfico de Coeficientes", "Odds Ratios", "Tabla de Coeficientes"])
                
                # Pestaña 1: Gráfico de Coeficientes
                with coef_tabs[0]:
                    # Gráfico de coeficientes ordenados por magnitud
                    model_results_plot = model_results.copy()
                    model_results_plot['abs_coef'] = model_results_plot['Coeficiente'].abs()
                    model_results_plot = model_results_plot.sort_values('abs_coef', ascending=True)
                    
                    # Formatear etiquetas
                    model_results_plot['Etiqueta'] = model_results_plot['Variable'].apply(lambda x: var_labels.get(x, x))
                    
                    # Crear un gráfico de barras horizontales con intervalos de confianza
                    fig_coef = go.Figure()
                    
                    # Añadir barras para coeficientes
                    fig_coef.add_trace(go.Bar(
                        y=model_results_plot['Etiqueta'],
                        x=model_results_plot['Coeficiente'],
                        orientation='h',
                        marker_color=[
                            '#e74c3c' if x < 0 and p < 0.05 else
                            '#2ecc71' if x > 0 and p < 0.05 else
                            '#f39c12' if x < 0 else '#3498db'
                            for x, p in zip(model_results_plot['Coeficiente'], model_results_plot['p-valor'])
                        ],
                        error_x=dict(
                            type='data',
                            array=model_results_plot['Error Estándar'] * 1.96,
                            visible=True
                        ),
                        name='Coeficiente',
                        customdata=np.stack((
                            model_results_plot['p-valor'],
                            model_results_plot['Significativo']
                        ), axis=-1),
                        hovertemplate='<b>%{y}</b><br>'
                                     'Coeficiente: %{x:.3f}<br>'
                                     'I.C. 95%%: [%{x}-1.96×%{error_x.array:.3f}, %{x}+1.96×%{error_x.array:.3f}]<br>'
                                     'p-valor: %{customdata[0]:.4f}<br>'
                                     'Significativo: %{customdata[1]}<br>'
                                     '<extra></extra>'
                    ))
                    
                    # Añadir línea vertical en cero
                    fig_coef.add_shape(
                        type="line",
                        line=dict(dash="dash", color="gray", width=1),
                        x0=0, x1=0,
                        y0=-0.5, y1=len(model_results_plot) - 0.5
                    )
                    
                    # Mejorar layout
                    fig_coef.update_layout(
                        title={
                            'text': 'Coeficientes del Modelo Ordinal Logístico',
                            'font': {'size': 20}
                        },
                        xaxis_title='Coeficiente (log-odds)',
                        yaxis_title='Variable',
                        height=600,
                        margin=dict(l=0, r=10, t=50, b=20)
                    )
                    
                    st.plotly_chart(fig_coef, use_container_width=True)
                    
                    # Interpretación del gráfico
                    st.markdown('<div class="explanation">', unsafe_allow_html=True)
                    st.markdown("""
                    **Interpretación:**
                    - Barras **verdes** indican un efecto positivo estadísticamente significativo (aumenta la felicidad)
                    - Barras **rojas** indican un efecto negativo estadísticamente significativo (disminuye la felicidad)
                    - Barras **azules/naranjas** indican efectos no estadísticamente significativos
                    - Las líneas horizontales representan el intervalo de confianza del 95% para cada coeficiente
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Pestaña 2: Odds Ratios
                with coef_tabs[1]:
                    # Gráfico de odds ratios (más intuitivo para interpretación)
                    odds_ratios_plot = odds_ratios.copy()
                    odds_ratios_plot['Etiqueta'] = odds_ratios_plot['Variable'].apply(lambda x: var_labels.get(x, x))
                    
                    # Ordenar por magnitud del odds ratio (distancia desde 1)
                    odds_ratios_plot['distance_from_1'] = (odds_ratios_plot['Odds Ratio'] - 1).abs()
                    odds_ratios_plot = odds_ratios_plot.sort_values('distance_from_1', ascending=True)
                    
                    # Crear gráfico de forest plot para odds ratios
                    fig_odds = go.Figure()
                    
                    # Añadir puntos para odds ratios con intervalos de confianza
                    fig_odds.add_trace(go.Scatter(
                        y=odds_ratios_plot['Etiqueta'],
                        x=odds_ratios_plot['Odds Ratio'],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=[
                                '#e74c3c' if x < 1 and p < 0.05 else
                                '#2ecc71' if x > 1 and p < 0.05 else
                                '#f39c12' if x < 1 else '#3498db'
                                for x, p in zip(odds_ratios_plot['Odds Ratio'], odds_ratios_plot['p-valor'])
                            ],
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        error_x=dict(
                            type='data',
                            array=odds_ratios_plot['OR Upper CI'] - odds_ratios_plot['Odds Ratio'],
                            arrayminus=odds_ratios_plot['Odds Ratio'] - odds_ratios_plot['OR Lower CI'],
                            visible=True
                        ),
                        name='Odds Ratio',
                        customdata=np.stack((
                            odds_ratios_plot['p-valor'],
                            odds_ratios_plot['Significativo'],
                            odds_ratios_plot['OR Lower CI'],
                            odds_ratios_plot['OR Upper CI']
                        ), axis=-1),
                        hovertemplate='<b>%{y}</b><br>'
                                     'Odds Ratio: %{x:.2f}<br>'
                                     'I.C. 95%%: [%{customdata[2]:.2f}, %{customdata[3]:.2f}]<br>'
                                     'p-valor: %{customdata[0]:.4f}<br>'
                                     'Significativo: %{customdata[1]}<br>'
                                     '<extra></extra>'
                    ))
                    
                    # Añadir línea vertical en 1 (sin efecto)
                    fig_odds.add_shape(
                        type="line",
                        line=dict(dash="dash", color="gray", width=1),
                        x0=1, x1=1,
                        y0=-0.5, y1=len(odds_ratios_plot) - 0.5
                    )
                    
                    # Mejorar layout y escala logarítmica para mejor visualización
                    fig_odds.update_layout(
                        title={
                            'text': 'Odds Ratios del Modelo (escala logarítmica)',
                            'font': {'size': 20}
                        },
                        xaxis_title='Odds Ratio (OR)',
                        yaxis_title='Variable',
                        xaxis_type="log",
                        height=600,
                        margin=dict(l=0, r=10, t=50, b=20)
                    )
                    
                    # Añadir escala para referencia
                    or_min = odds_ratios_plot['OR Lower CI'].min()
                    or_max = odds_ratios_plot['OR Upper CI'].max()
                    
                    # Asegurar que la escala incluya el 1
                    xrange_min = min(or_min, 0.5)
                    xrange_max = max(or_max, 2.0)
                    
                    fig_odds.update_xaxes(range=[np.log10(xrange_min), np.log10(xrange_max)])
                    
                    # Añadir guías verticales para ayudar en la interpretación
                    for ratio in [0.5, 0.75, 1.25, 1.5, 2.0]:
                        if xrange_min <= ratio <= xrange_max:
                            fig_odds.add_shape(
                                type="line",
                                line=dict(dash="dot", color="lightgray", width=1),
                                x0=ratio, x1=ratio,
                                y0=-0.5, y1=len(odds_ratios_plot) - 0.5
                            )
                    
                    st.plotly_chart(fig_odds, use_container_width=True)
                    
                    # Interpretación del gráfico
                    st.markdown('<div class="explanation">', unsafe_allow_html=True)
                    st.markdown("""
                    **Interpretación de los Odds Ratios:**
                    - OR > 1: La variable aumenta la probabilidad de niveles más altos de felicidad
                    - OR < 1: La variable disminuye la probabilidad de niveles más altos de felicidad
                    - OR = 1: La variable no tiene efecto en la felicidad
                    - Los puntos **verdes** y **rojos** indican efectos estadísticamente significativos
                    - Las líneas horizontales representan el intervalo de confianza del 95% para cada odds ratio
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Explicación sobre cómo interpretar los odds ratios
                    with st.expander("💡 ¿Cómo interpretar un Odds Ratio?"):
                        st.markdown("""
                        Un **Odds Ratio (OR)** es una forma de expresar la asociación entre una variable y un resultado. En este caso, indica cómo cambia la probabilidad de tener mayor felicidad cuando la variable cambia.

                        - **OR = 2.0**: La probabilidad de mayor felicidad se duplica cuando la variable aumenta en 1 unidad
                        - **OR = 1.5**: La probabilidad de mayor felicidad aumenta un 50% cuando la variable aumenta en 1 unidad
                        - **OR = 1.0**: No hay efecto (línea de referencia)
                        - **OR = 0.67**: La probabilidad de mayor felicidad disminuye un 33% cuando la variable aumenta en 1 unidad
                        - **OR = 0.5**: La probabilidad de mayor felicidad se reduce a la mitad cuando la variable aumenta en 1 unidad

                        Para variables categóricas (como género o víctima), el OR compara una categoría con la otra.
                        """)
                
                # Pestaña 3: Tabla de Coeficientes
                with coef_tabs[2]:
                    # Tabla detallada de coeficientes
                    st.subheader("Tabla de Coeficientes y Significancia")
                    
                    # Preparar datos para mostrar
                    display_results = model_results.copy()
                    display_results['Magnitud'] = display_results['Coeficiente'].abs()
                    display_results = display_results.sort_values(['Significativo', 'Magnitud'], ascending=[False, False])
                    
                    # Añadir etiquetas amigables para las variables
                    display_results['Etiqueta'] = display_results['Variable'].apply(lambda x: var_labels.get(x, x))
                    
                    # Formatear para mostrar
                    display_results['Coeficiente'] = display_results['Coeficiente'].round(3)
                    display_results['Error Estándar'] = display_results['Error Estándar'].round(3)
                    display_results['p-valor'] = display_results['p-valor'].round(4)
                    display_results['OR'] = np.exp(display_results['Coeficiente']).round(3)
                    
                    # Mostrar solo las columnas relevantes
                    st.dataframe(
                        display_results[['Variable', 'Etiqueta', 'Coeficiente', 'Error Estándar', 'p-valor', 'OR', 'Significativo']],
                        column_config={
                            "Variable": "Código",
                            "Etiqueta": "Variable",
                            "Coeficiente": st.column_config.NumberColumn("Coeficiente", format="%.3f"),
                            "Error Estándar": st.column_config.NumberColumn("Error Estándar", format="%.3f"),
                            "p-valor": st.column_config.NumberColumn("p-valor", format="%.4f"),
                            "OR": st.column_config.NumberColumn("Odds Ratio", format="%.3f"),
                            "Significativo": st.column_config.CheckboxColumn("Significativo (p<0.05)")
                        },
                        use_container_width=True
                    )
                
                # Interpretación del modelo
                st.subheader("🧩 Interpretación de los Resultados")
                
                # Ordenar variables por importancia (magnitud y significancia)
                significant_vars = model_results[model_results['Significativo']]
                significant_pos = significant_vars[significant_vars['Coeficiente'] > 0].sort_values('Coeficiente', ascending=False)
                significant_neg = significant_vars[significant_vars['Coeficiente'] < 0].sort_values('Coeficiente')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Factores que Aumentan la Felicidad")
                    
                    if len(significant_pos) > 0:
                        for i, (_, row) in enumerate(significant_pos.iterrows()):
                            var_name = var_labels.get(row['Variable'], row['Variable'])
                            effect_size = "fuerte" if abs(row['Coeficiente']) > 0.3 else "moderado" if abs(row['Coeficiente']) > 0.1 else "leve"
                            odds = np.exp(row['Coeficiente'])
                            
                            # Emoji dependiendo de la magnitud
                            emoji = "🔥" if abs(row['Coeficiente']) > 0.3 else "⬆️" if abs(row['Coeficiente']) > 0.1 else "➕"
                            
                            st.markdown(f"""
                            <div style="
                                padding: 10px; 
                                background-color: rgba(46, 204, 113, 0.1); 
                                border-radius: 10px; 
                                margin-bottom: 10px;
                                border-left: 5px solid #2ecc71;
                            ">
                                <h4 style="margin-top: 0;">{emoji} {var_name}</h4>
                                <p><b>Coeficiente:</b> +{row['Coeficiente']:.3f}
                                <span style="float: right;"><b>OR:</b> {odds:.2f}</span></p>
                                <p>Efecto {effect_size} positivo sobre la felicidad.</p>
                                <p>Aumentar {row['Variable']} en 1 unidad multiplica por <b>{odds:.2f}</b> las probabilidades de mayor felicidad.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No se encontraron factores con efecto positivo significativo.")
                
                with col2:
                    st.markdown("### 📉 Factores que Disminuyen la Felicidad")
                    
                    if len(significant_neg) > 0:
                        for i, (_, row) in enumerate(significant_neg.iterrows()):
                            var_name = var_labels.get(row['Variable'], row['Variable'])
                            effect_size = "fuerte" if abs(row['Coeficiente']) > 0.3 else "moderado" if abs(row['Coeficiente']) > 0.1 else "leve"
                            odds = np.exp(row['Coeficiente'])
                            
                            # Emoji dependiendo de la magnitud
                            emoji = "💥" if abs(row['Coeficiente']) > 0.3 else "⬇️" if abs(row['Coeficiente']) > 0.1 else "➖"
                            
                            st.markdown(f"""
                            <div style="
                                padding: 10px; 
                                background-color: rgba(231, 76, 60, 0.1); 
                                border-radius: 10px; 
                                margin-bottom: 10px;
                                border-left: 5px solid #e74c3c;
                            ">
                                <h4 style="margin-top: 0;">{emoji} {var_name}</h4>
                                <p><b>Coeficiente:</b> {row['Coeficiente']:.3f}
                                <span style="float: right;"><b>OR:</b> {odds:.2f}</span></p>
                                <p>Efecto {effect_size} negativo sobre la felicidad.</p>
                                <p>Aumentar {row['Variable']} en 1 unidad multiplica por <b>{odds:.2f}</b> las probabilidades de mayor felicidad.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No se encontraron factores con efecto negativo significativo.")
                
                # Resumen general
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.markdown("### 📝 Conclusiones del Modelo")
                
                # Generar un resumen del modelo basado en los resultados
                summary_points = []
                
                # Variables más importantes
                if len(significant_pos) > 0:
                    top_pos = significant_pos.iloc[0]
                    var_name = var_labels.get(top_pos['Variable'], top_pos['Variable'])
                    summary_points.append(f"El factor con mayor efecto positivo es **{var_name}** (OR = {np.exp(top_pos['Coeficiente']):.2f}).")
                
                if len(significant_neg) > 0:
                    top_neg = significant_neg.iloc[0]
                    var_name = var_labels.get(top_neg['Variable'], top_neg['Variable'])
                    summary_points.append(f"El factor con mayor efecto negativo es **{var_name}** (OR = {np.exp(top_neg['Coeficiente']):.2f}).")
                
                # Categorías generales
                health_vars = [v for v in significant_vars['Variable'] if v in ['health']]
                economic_vars = [v for v in significant_vars['Variable'] if v in ['fin_sat']]
                social_vars = [v for v in significant_vars['Variable'] if v in ['rel_import', 'imp_family', 'imp_friends']]
                security_vars = [v for v in significant_vars['Variable'] if v in ['sec_rec', 'victim', 'indice_criminalidad']]
                
                if health_vars:
                    summary_points.append("Los factores relacionados con la **salud** son importantes determinantes de la felicidad.")
                
                if economic_vars:
                    summary_points.append("La **situación económica** tiene un impacto significativo en la felicidad.")
                
                if social_vars:
                    summary_points.append("Las **relaciones sociales** juegan un papel crucial en la felicidad.")
                
                if security_vars:
                    summary_points.append("La **seguridad y el entorno** influyen significativamente en la felicidad.")
                
                # Mostrar puntos del resumen
                for point in summary_points:
                    st.markdown(f"- {point}")
                
                st.markdown("""
                Este modelo multivariado muestra cómo distintas variables influyen en la probabilidad de tener mayores niveles de felicidad, controlando por el efecto de las demás variables. Los coeficientes representan el efecto ajustado de cada variable, lo que permite identificar los factores más importantes para la felicidad.
                """)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Perfil idealizado
                with st.expander("👤 Perfil de Persona más Feliz"):
                    st.markdown("### Características asociadas a mayor felicidad")
                    
                    # Generar perfil basado en los coeficientes del modelo
                    happy_profile = generate_happy_profile(model_results, df, model_vars, var_labels)
                    
                    st.dataframe(
                        happy_profile,
                        column_config={
                            "Variable": st.column_config.TextColumn("Característica"),
                            "Valor Ideal": st.column_config.TextColumn("Valor Ideal"),
                        },
                        use_container_width=True
                    )
                    
                    st.markdown("""
                    **Nota:** Este perfil está basado en los coeficientes del modelo y representa las características que estadísticamente están asociadas con mayores niveles de felicidad. Para variables con efecto positivo se eligieron valores altos, y para variables con efecto negativo se eligieron valores bajos.
                    """)
            except Exception as e:
                st.error(f"Error al ajustar el modelo: {str(e)}")
                st.info("Esto puede deberse a problemas de convergencia o multicolinealidad en los datos. Pruebe con un conjunto diferente de variables predictoras.")
    else:
        st.warning(f"No hay suficientes datos para ajustar el modelo. Se necesitan al menos {len(model_vars) + 10} observaciones completas.")

# TAB 4: SIMULADOR DE PREDICCIONES
with tab4:
    colored_header(
        label="Simulador de Felicidad",
        description="Ajusta las variables para ver cómo cambian las predicciones de felicidad",
        color_name="blue-70"
    )
    
    st.markdown("""
    <div class="info-box">
    <h4>🧠 ¿Cómo funciona el simulador?</h4>
    <p>Este simulador te permite explorar cómo diferentes factores afectan la probabilidad de felicidad:</p>
    <ul>
        <li>Ajusta las variables usando los controles de la izquierda</li>
        <li>Observa cómo cambian las probabilidades de cada nivel de felicidad en tiempo real</li>
        <li>Analiza la sensibilidad de la felicidad a cambios en una variable específica</li>
        <li>Crea perfiles personalizados y compara diferentes escenarios</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar si hay un modelo ajustado
    try:
        # Intentar acceder a model para ver si está definido
        _ = model.params
        
        # Valores por defecto (medianas)
        default_values = {var: float(df[var].dropna().median()) for var in model_vars}
        
        # Layout con dos columnas: controles y gráfico
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Ajustar Variables")
            
            # Crear sliders para cada variable y almacenar valores
            adjusted_values = {}
            
            # Organizar variables en grupos con expanders
            with st.expander("👤 Variables Demográficas", expanded=True):
                age = st.slider(
                    "Edad", 
                    int(df['age'].dropna().min()), 
                    int(df['age'].dropna().max()), 
                    int(default_values['age']),
                    key="sim_age",
                    help=var_descriptions.get('age', '')
                )
                adjusted_values['age'] = age
                
                female = st.radio(
                    "Género", 
                    options=[0, 1], 
                    format_func=lambda x: gender_labels[x],
                    index=int(default_values['female']),
                    key="sim_female",
                    help=var_descriptions.get('female', '')
                )
                adjusted_values['female'] = female
                
                soc_class = st.select_slider(
                    "Clase Social", 
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: class_labels[int(x)-1],
                    value=int(default_values['soc_class']),
                    key="sim_soc_class",
                    help=var_descriptions.get('soc_class', '')
                )
                adjusted_values['soc_class'] = soc_class
            
            with st.expander("💪 Bienestar y Satisfacción", expanded=True):
                health = st.select_slider(
                    "Salud", 
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: ["Muy mala", "Mala", "Regular", "Buena", "Muy buena"][x-1],
                    value=int(default_values['health']),
                    key="sim_health",
                    help=var_descriptions.get('health', '')
                )
                adjusted_values['health'] = health
                
                life_sat = st.slider(
                    "Satisfacción con la Vida", 
                    1, 10, int(default_values['life_sat']),
                    key="sim_life_sat",
                    help=var_descriptions.get('life_sat', '')
                )
                adjusted_values['life_sat'] = life_sat
                
                fin_sat = st.slider(
                    "Satisfacción Financiera", 
                    1, 10, int(default_values['fin_sat']),
                    key="sim_fin_sat",
                    help=var_descriptions.get('fin_sat', '')
                )
                adjusted_values['fin_sat'] = fin_sat
            
            with st.expander("🧠 Valores y Actitudes", expanded=False):
                god_imp = st.slider(
                    "Importancia de Dios", 
                    1, 10, int(default_values['god_imp']),
                    key="sim_god_imp",
                    help=var_descriptions.get('god_imp', '')
                )
                adjusted_values['god_imp'] = god_imp
                
                control = st.slider(
                    "Control sobre la Vida", 
                    1, 10, int(default_values['control']),
                    key="sim_control",
                    help=var_descriptions.get('control', '')
                )
                adjusted_values['control'] = control
                
                rel_import = st.slider(
                    "Importancia de Relaciones", 
                    1.0, 4.0, float(default_values['rel_import']), 0.1,
                    key="sim_rel_import",
                    help=var_descriptions.get('rel_import', '')
                )
                adjusted_values['rel_import'] = rel_import
                
                sec_rec = st.select_slider(
                    "Seguridad Residencial", 
                    options=[1, 2, 3, 4],
                    format_func=lambda x: ["Muy inseguro", "Inseguro", "Seguro", "Muy seguro"][x-1],
                    value=int(default_values['sec_rec']),
                    key="sim_sec_rec",
                    help=var_descriptions.get('sec_rec', '')
                )
                adjusted_values['sec_rec'] = sec_rec
            
            with st.expander("🏙️ Contexto Social", expanded=False):
                indice_criminalidad = st.slider(
                    "Índice de Criminalidad", 
                    float(df['indice_criminalidad'].dropna().min()), 
                    float(df['indice_criminalidad'].dropna().max()), 
                    float(default_values['indice_criminalidad']), 0.1,
                    key="sim_indice_criminalidad",
                    help=var_descriptions.get('indice_criminalidad', '')
                )
                adjusted_values['indice_criminalidad'] = indice_criminalidad
                
                victim = st.radio(
                    "Víctima de Delito", 
                    options=[0, 1], 
                    format_func=lambda x: victim_labels[x],
                    index=int(default_values['victim']),
                    key="sim_victim",
                    help=var_descriptions.get('victim', '')
                )
                adjusted_values['victim'] = victim
            
            # Botones para perfiles predefinidos
            st.subheader("👥 Perfiles Predefinidos")
            
            profiles_cols = st.columns(2)
            
            with profiles_cols[0]:
                if st.button("🔄 Valores Medianos", help="Resetear todos los valores a las medianas de la población"):
                    st.rerun()
            
            with profiles_cols[1]:
                if st.button("✨ Perfil Optimizado", help="Configurar las variables para maximizar la felicidad según el modelo"):
                    # Los valores se establecerán en la siguiente ejecución
                    # Usamos cache para almacenar temporalmente los valores óptimos
                    st.session_state.optimal_profile = True
                    st.rerun()
        
        with col2:
            st.subheader("📊 Predicciones de Felicidad")
            
            # Verificar si debemos cargar el perfil optimizado
            if 'optimal_profile' in st.session_state and st.session_state.optimal_profile:
                st.info("Aplicando perfil optimizado para maximizar la felicidad...")
                
                # Generar perfil feliz basado en los coeficientes
                happy_profile_df = generate_happy_profile(model_results, df, model_vars, var_labels)
                happy_profile = {}
                
                # Buscar correspondencia entre las variables y los valores ideales
                for var in model_vars:
                    # Encontrar el índice correspondiente
                    idx = happy_profile_df[happy_profile_df['Variable'] == var_labels.get(var, var)].index
                    
                    if len(idx) > 0:
                        val_str = happy_profile_df.loc[idx[0], 'Valor Ideal']
                        
                        # Convertir a valores numéricos
                        if var in ['female', 'victim']:
                            if var == 'female':
                                val = 1 if val_str == 'Mujer' else 0
                            else:
                                val = 1 if val_str == 'Sí' else 0
                        elif var == 'soc_class':
                            try:
                                # Buscar el índice del valor en class_labels
                                val = class_labels.index(val_str) + 1
                            except ValueError:
                                # Si no se encuentra, usar el valor predeterminado
                                val = default_values[var]
                        else:
                            try:
                                val = float(val_str)
                            except ValueError:
                                val = default_values[var]
                    else:
                        val = default_values[var]
                    
                    adjusted_values[var] = val
                
                # Resetear el flag para la próxima iteración
                st.session_state.optimal_profile = False
            
            # Crear un DataFrame con los valores ajustados
            pred_df = pd.DataFrame([adjusted_values])
            
            try:
                # Generar predicciones
                predictions = model.model.predict(model.params, exog=pred_df)[0]
                
                # Crear gráfico de probabilidades más atractivo
                labels = happiness_labels
                colors = ['#e74c3c', '#f39c12', '#2ecc71', '#27ae60']
                
                # Gráfico de barras horizontales con diseño mejorado
                fig_pred = go.Figure()
                
                for i, (label, prob, color) in enumerate(zip(labels, predictions, colors)):
                    fig_pred.add_trace(go.Bar(
                        y=[label],
                        x=[prob * 100],
                        orientation='h',
                        name=label,
                        marker_color=color,
                        text=[f"{prob * 100:.1f}%"],
                        textposition='auto',
                        hoverinfo='text',
                        hovertext=[f"{label}: {prob*100:.1f}%"]
                    ))
                
                fig_pred.update_layout(
                    title={
                        'text': 'Probabilidad de Cada Nivel de Felicidad',
                        'font': {'size': 18}
                    },
                    xaxis_title='Probabilidad (%)',
                    yaxis_title='Nivel de Felicidad',
                    xaxis=dict(range=[0, 100]),
                    height=300,
                    margin=dict(l=0, r=0, t=50, b=30),
                    barmode='group',
                    yaxis=dict(
                        categoryorder='array',
                        categoryarray=['Muy Infeliz', 'Infeliz', 'Feliz', 'Muy Feliz']
                    ),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Mostrar el nivel más probable con diseño mejorado
                max_prob_index = np.argmax(predictions)
                max_prob_label = labels[max_prob_index]
                max_prob_value = predictions[max_prob_index]
                
                st.markdown("### Resultado más probable:")
                
                # Estilo para el resultado
                result_color = colors[max_prob_index]
                result_emoji = "😢" if max_prob_index == 0 else "😔" if max_prob_index == 1 else "😊" if max_prob_index == 2 else "😄"
                
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(90deg, {result_color}33, {result_color}11); padding: 20px; border-radius: 10px; text-align: center; border-left: 8px solid {result_color};">
                        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">{result_emoji} {max_prob_label}</h1>
                        <h3 style="opacity: 0.8;">Probabilidad: {max_prob_value*100:.1f}%</h3>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Mostrar todas las probabilidades en formato de tabla
                results_df = pd.DataFrame({
                    'Nivel': labels,
                    'Probabilidad': [f"{p*100:.1f}%" for p in predictions]
                })
                
                with st.expander("📊 Ver todas las probabilidades"):
                    st.dataframe(
                        results_df,
                        column_config={
                            "Nivel": st.column_config.TextColumn("Nivel de Felicidad"),
                            "Probabilidad": st.column_config.TextColumn("Probabilidad"),
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Gráfico adicional: variable focal
                st.subheader("📈 Análisis de Sensibilidad")
                
                # Seleccionar variable focal
                focal_var = st.selectbox(
                    "Selecciona una variable para análisis de sensibilidad:", 
                    model_vars,
                    format_func=lambda x: var_labels.get(x, x),
                    index=model_vars.index('fin_sat'),
                    key="sim_focal_var"
                )
                
                # Determinar el rango para la variable focal
                if focal_var in ['female', 'victim']:
                    # Para variables binarias solo hay dos valores
                    var_min, var_max = 0, 1
                    num_points = 2
                else:
                    # Para variables continuas, usar rango real
                    var_min = float(df[focal_var].dropna().min())
                    var_max = float(df[focal_var].dropna().max())
                    num_points = 10
                
                # Generar predicciones marginales
                focal_values, marginal_preds = generate_marginal_predictions(
                    model.model, model, adjusted_values, focal_var, var_min, var_max, num_points)
                
                # Crear gráfico de líneas más atractivo
                fig_marginal = go.Figure()
                
                # Determinar qué trace mostrar activamente
                active_trace = 2 if max_prob_index <= 1 else 3  # Mostrar "Feliz" o "Muy Feliz" dependiendo del resultado actual
                
                for i, label in enumerate(labels):
                    fig_marginal.add_trace(go.Scatter(
                        x=focal_values,
                        y=marginal_preds[:, i] * 100,
                        mode='lines+markers',
                        name=label,
                        line=dict(
                            color=colors[i], 
                            width=3 if i == active_trace else 2,
                            dash='solid' if i == active_trace else 'dash'
                        ),
                        marker=dict(
                            size=8 if i == active_trace else 6
                        ),
                        opacity=1.0 if i == active_trace else 0.7
                    ))
                
                # Añadir línea vertical en el valor actual
                current_value = adjusted_values[focal_var]
                
                fig_marginal.add_vline(
                    x=current_value,
                    line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dash"),
                    annotation_text="Valor Actual"
                )
                
                # Estilizar el gráfico
                var_label = var_labels.get(focal_var, focal_var)
                
                # Formatear el eje X para variables categóricas
                if focal_var == 'female':
                    tick_vals = [0, 1]
                    tick_text = ['Hombre', 'Mujer']
                elif focal_var == 'victim':
                    tick_vals = [0, 1]
                    tick_text = ['No', 'Sí']
                elif focal_var == 'soc_class':
                    tick_vals = [1, 2, 3, 4, 5]
                    tick_text = class_labels
                else:
                    tick_vals = None
                    tick_text = None
                
                # Actualizar layout
                fig_marginal.update_layout(
                    title={
                        'text': f'Efecto de {var_label} en la Probabilidad de Felicidad',
                        'font': {'size': 18}
                    },
                    xaxis_title=var_label,
                    yaxis_title='Probabilidad (%)',
                    height=400,
                    margin=dict(l=0, r=0, t=50, b=30),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    hovermode='x unified'
                )
                
                # Actualizar ejes
                if tick_vals is not None and tick_text is not None:
                    fig_marginal.update_xaxes(
                        tickmode='array',
                        tickvals=tick_vals,
                        ticktext=tick_text
                    )
                
                st.plotly_chart(fig_marginal, use_container_width=True)
                
                # Explicación del gráfico
                st.markdown('<div class="explanation">', unsafe_allow_html=True)
                st.markdown(f"""
                **Interpretación:**
                Este gráfico muestra cómo cambia la probabilidad de cada nivel de felicidad al variar **{var_label}**, 
                manteniendo todas las demás variables constantes. La línea vertical indica el valor actual seleccionado.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Comparación con el perfil promedio
                with st.expander("📊 Comparación con Perfil Promedio"):
                    # Crear un perfil con valores medios
                    average_values = {var: float(df[var].dropna().median()) for var in model_vars}
                    avg_df = pd.DataFrame([average_values])
                    
                    # Predecir para perfil promedio
                    avg_predictions = model.model.predict(model.params, exog=avg_df)[0]
                    
                    # Crear DataFrame para comparar
                    compare_df = pd.DataFrame({
                        'Nivel': labels,
                        'Tu Perfil': [f"{p*100:.1f}%" for p in predictions],
                        'Perfil Promedio': [f"{p*100:.1f}%" for p in avg_predictions],
                    })
                    
                    st.dataframe(
                        compare_df,
                        column_config={
                            "Nivel": st.column_config.TextColumn("Nivel de Felicidad"),
                            "Tu Perfil": st.column_config.TextColumn("Tu Perfil"),
                            "Perfil Promedio": st.column_config.TextColumn("Perfil Promedio"),
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Gráfico de comparación
                    compare_data = pd.DataFrame({
                        'Nivel': np.repeat(labels, 2),
                        'Perfil': ['Tu Perfil']*4 + ['Perfil Promedio']*4,
                        'Probabilidad': np.concatenate([predictions*100, avg_predictions*100])
                    })
                    
                    fig_compare = px.bar(
                        compare_data,
                        x='Nivel',
                        y='Probabilidad',
                        color='Perfil',
                        barmode='group',
                        text_auto='.1f',
                        labels={'Probabilidad': 'Probabilidad (%)'},
                        title='Comparación de Probabilidades',
                        color_discrete_map={
                            'Tu Perfil': '#3498db',
                            'Perfil Promedio': '#95a5a6'
                        }
                    )
                    
                    fig_compare.update_layout(
                        xaxis=dict(categoryorder='array', categoryarray=labels),
                        yaxis=dict(range=[0, 100]),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    # Calcular diferencias
                    diff = predictions - avg_predictions
                    max_diff_idx = np.argmax(np.abs(diff))
                    
                    if np.abs(diff[max_diff_idx]) > 0.05:  # Diferencia mayor al 5%
                        st.markdown(f"""
                        **Observación Principal:**
                        Tu perfil muestra una diferencia de **{diff[max_diff_idx]*100:.1f}%** en la probabilidad de ser **{labels[max_diff_idx]}** 
                        comparado con el perfil promedio.
                        """)
            
            except Exception as e:
                st.error(f"Error al generar predicciones: {str(e)}")
    except NameError:
        st.warning("No hay un modelo ajustado disponible. Primero debe ir a la pestaña 'Modelo' para ajustar el modelo.")
        
        # Mostrar una simulación básica
        st.markdown("""
        ### 🛠️ Simulación no disponible
        
        Para utilizar el simulador, primero debe ir a la pestaña 'Modelo' y esperar a que se ajuste el modelo estadístico.
        
        Una vez que el modelo esté disponible, podrá:
        - Ajustar diferentes variables y ver cómo afectan a la felicidad
        - Explorar diferentes perfiles y escenarios
        - Realizar análisis de sensibilidad
        """)

# TAB 5: ANÁLISIS DE INTERACCIONES
with tab5:
    colored_header(
        label="Análisis de Interacciones",
        description="Explora cómo diferentes variables interactúan entre sí para afectar la felicidad",
        color_name="blue-70"
    )
    
    st.markdown("""
    <div class="info-box">
    <h4>🧠 ¿Qué son las interacciones?</h4>
    <p>Una <b>interacción</b> ocurre cuando el efecto de una variable sobre la felicidad depende del valor de otra variable. Por ejemplo:</p>
    <ul>
        <li>El efecto de la satisfacción financiera podría ser más fuerte en personas con mala salud que en personas con buena salud.</li>
        <li>La importancia de las relaciones sociales podría tener un efecto diferente según el género.</li>
    </ul>
    <p>Este análisis nos permite detectar efectos conjuntos que no serían evidentes al analizar las variables individualmente.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar si hay un modelo ajustado
    try:
        # Intentar acceder a model para ver si está definido
        _ = model.params
        
        # Mostrar opciones para seleccionar variables
        st.subheader("🔍 Seleccionar Variables para la Interacción")
        
        # Layout con dos columnas para seleccionar variables
        col1, col2 = st.columns(2)
        
        with col1:
            var1 = st.selectbox(
                "Primera Variable:", 
                model_vars, 
                format_func=lambda x: var_labels.get(x, x),
                index=model_vars.index('health'),
                key="interaction_var1"
            )
            
            # Mostrar descripción de la variable
            if var1 in var_descriptions:
                st.markdown(f'<p class="small-font">{var_descriptions[var1]}</p>', unsafe_allow_html=True)
        
        with col2:
            # Filtrar para no seleccionar la misma variable
            var2_options = [v for v in model_vars if v != var1]
            var2_index = 0  # Por defecto, tomar el primer elemento
            
            var2 = st.selectbox(
                "Segunda Variable:", 
                var2_options,
                format_func=lambda x: var_labels.get(x, x),
                index=var2_index,
                key="interaction_var2"
            )
            
            # Mostrar descripción de la variable
            if var2 in var_descriptions:
                st.markdown(f'<p class="small-font">{var_descriptions[var2]}</p>', unsafe_allow_html=True)
                
        # Crear nombre de interacción
        interaction_name = f"{var1}_x_{var2}"
        
        try:
            # Datos para el modelo
            interaction_df = df[model_vars + ['happiness']].dropna()
            
            # Verificar si hay suficientes datos
            if len(interaction_df) > len(model_vars) + 10:
                with st.spinner('Ajustando modelo con interacción...'):
                    # Crear modelo con interacción
                    interaction_X = interaction_df[model_vars].copy()
                    interaction_X[interaction_name] = interaction_X[var1] * interaction_X[var2]
                    
                    interaction_mod = OrderedModel(interaction_df['happiness'], interaction_X, distr='logit')
                    interaction_res = interaction_mod.fit(method='bfgs', disp=False)
                    
                    # Extraer coeficientes
                    interaction_results = pd.DataFrame({
                        'Variable': interaction_res.params.index,
                        'Coeficiente': interaction_res.params.values,
                        'Error Estándar': interaction_res.bse.values,
                        'p-valor': interaction_res.pvalues.values,
                        'Significativo': interaction_res.pvalues < 0.05
                    })
                
                # Mostrar resultados de la interacción con mejor diseño
                st.subheader("📊 Resultados de la Interacción")
                
                # Layout con dos columnas
                col_main, col_side = st.columns([2, 1])
                
                with col_main:
                    # Enfocarse en las variables de interés y la interacción
                    focus_vars = [var1, var2, interaction_name]
                    focus_results = interaction_results[interaction_results['Variable'].isin(focus_vars)]
                    
                    # Verificar que tenemos resultados para las variables de interés
                    if len(focus_results) == 3:
                        # Añadir etiquetas amigables para las variables
                        focus_results['Etiqueta'] = focus_results['Variable'].apply(
                            lambda x: x if x == interaction_name else var_labels.get(x, x))
                        
                        # Formatear la etiqueta de interacción
                        interaction_idx = focus_results[focus_results['Variable'] == interaction_name].index
                        if len(interaction_idx) > 0:
                            var1_label = var_labels.get(var1, var1)
                            var2_label = var_labels.get(var2, var2)
                            focus_results.loc[interaction_idx, 'Etiqueta'] = f"Interacción ({var1_label} × {var2_label})"
                        
                        # Gráfico de coeficientes para las variables de interés
                        fig_interaction = go.Figure()
                        
                        # Obtener el color para la interacción
                        interaction_coef = focus_results.loc[focus_results['Variable']==interaction_name, 'Coeficiente'].values[0]
                        interaction_pval = focus_results.loc[focus_results['Variable']==interaction_name, 'p-valor'].values[0]
                        
                        # Definir colores basados en significancia y dirección
                        bar_colors = []
                        for _, row in focus_results.iterrows():
                            if row['Variable'] == interaction_name:
                                # Interacción
                                if row['p-valor'] < 0.05:
                                    # Significativa
                                    color = '#e74c3c' if row['Coeficiente'] < 0 else '#2ecc71'
                                else:
                                    # No significativa
                                    color = '#f39c12' if row['Coeficiente'] < 0 else '#3498db'
                            else:
                                # Variables principales
                                color = '#3498db'
                            
                            bar_colors.append(color)
                        
                        # Añadir barras para coeficientes
                        fig_interaction.add_trace(go.Bar(
                            y=focus_results['Etiqueta'],
                            x=focus_results['Coeficiente'],
                            orientation='h',
                            marker_color=bar_colors,
                            error_x=dict(
                                type='data',
                                array=focus_results['Error Estándar'] * 1.96,
                                visible=True
                            )
                        ))
                        
                        # Añadir línea vertical en cero
                        fig_interaction.add_shape(
                            type="line",
                            line=dict(dash="dash", color="gray", width=1),
                            x0=0, x1=0,
                            y0=-0.5, y1=len(focus_results) - 0.5
                        )
                        
                        # Mejorar layout
                        fig_interaction.update_layout(
                            title=f'Coeficientes del Modelo con Interacción {var_labels.get(var1, var1)} × {var_labels.get(var2, var2)}',
                            xaxis_title='Coeficiente (log-odds)',
                            yaxis_title='Variable',
                            height=300,
                            margin=dict(l=0, r=0, t=50, b=30),
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_interaction, use_container_width=True)
                        
                        # Visualizaciones dependiendo del tipo de variables
                        if var1 not in ['female', 'victim'] and var2 not in ['female', 'victim']:
                            # Para dos variables continuas, usar mapa de calor
                            # Valores medios para las demás variables
                            default_values = {var: float(df[var].dropna().median()) for var in model_vars}
                            
                            # Crear una cuadrícula de valores para las dos variables
                            var1_values = np.linspace(df[var1].dropna().min(), df[var1].dropna().max(), 20)
                            var2_values = np.linspace(df[var2].dropna().min(), df[var2].dropna().max(), 20)
                            
                            # Crear malla
                            var1_grid, var2_grid = np.meshgrid(var1_values, var2_values)
                            
                            # Valores para predicción
                            grid_df = pd.DataFrame({
                                var1: var1_grid.flatten(),
                                var2: var2_grid.flatten()
                            })
                            
                            # Añadir valores por defecto para otras variables
                            for var in model_vars:
                                if var not in [var1, var2]:
                                    grid_df[var] = default_values[var]
                            
                            # Añadir interacción
                            grid_df[interaction_name] = grid_df[var1] * grid_df[var2]
                            
                            # Hacer predicciones
                            preds = interaction_mod.predict(interaction_res.params, exog=grid_df)
                            
                            # Índice del nivel de felicidad a mostrar (por defecto "Muy Feliz")
                            happiness_level = st.radio(
                                "Nivel de felicidad a visualizar:",
                                happiness_labels,
                                index=3,  # Muy Feliz por defecto
                                horizontal=True,
                                key="heatmap_happiness_level"
                            )
                            level_idx = happiness_labels.index(happiness_level)
                            
                            # Probabilidad del nivel seleccionado
                            prob_values = preds[:, level_idx].reshape(20, 20) * 100
                            
                            # Crear mapa de calor mejorado
                            fig_heatmap = go.Figure(data=go.Heatmap(
                                z=prob_values,
                                x=var1_values,
                                y=var2_values,
                                colorscale='Viridis',
                                colorbar=dict(
                                    title=f'Probabilidad de<br>{happiness_level} (%)',
                                    titleside='right'
                                ),
                                hovertemplate=(
                                    f"{var_labels.get(var1, var1)}: %{{x:.2f}}<br>" +
                                    f"{var_labels.get(var2, var2)}: %{{y:.2f}}<br>" +
                                    f"Probabilidad: %{{z:.1f}}%<extra></extra>"
                                )
                            ))
                            
                            # Añadir contornos para mayor claridad
                            fig_heatmap.add_trace(go.Contour(
                                z=prob_values,
                                x=var1_values,
                                y=var2_values,
                                showscale=False,
                                contours=dict(
                                    showlabels=True,
                                    labelfont=dict(size=10, color='white')
                                ),
                                line=dict(width=0.5, color='white'),
                                colorscale='Viridis'
                            ))
                            
                            # Marcar valores medianos con una cruz
                            fig_heatmap.add_trace(go.Scatter(
                                x=[default_values[var1]],
                                y=[default_values[var2]],
                                mode='markers',
                                marker=dict(
                                    symbol='cross',
                                    size=12,
                                    color='white',
                                    line=dict(width=1, color='black')
                                ),
                                name='Valores Medios'
                            ))
                            
                            # Mejorar layout
                            fig_heatmap.update_layout(
                                title=f'Probabilidad de {happiness_level} según {var_labels.get(var1, var1)} y {var_labels.get(var2, var2)}',
                                xaxis_title=var_labels.get(var1, var1),
                                yaxis_title=var_labels.get(var2, var2),
                                height=500,
                                margin=dict(l=0, r=0, t=50, b=30)
                            )
                            
                            # Determinar si hay interacción significativa
                            if interaction_pval < 0.05:
                                # Añadir anotación señalando la interacción
                                if interaction_coef > 0:
                                    annotation_text = "Interacción positiva significativa"
                                    annotation_color = "rgba(46, 204, 113, 0.8)"
                                else:
                                    annotation_text = "Interacción negativa significativa"
                                    annotation_color = "rgba(231, 76, 60, 0.8)"
                                
                                fig_heatmap.add_annotation(
                                    xref="paper", yref="paper",
                                    x=0.5, y=1.06,
                                    text=annotation_text,
                                    showarrow=False,
                                    font=dict(color="white", size=12),
                                    bgcolor=annotation_color,
                                    bordercolor="rgba(0,0,0,0)",
                                    borderwidth=0,
                                    borderpad=4,
                                    width=220,
                                    # borderradius=5,
                                    align="center"
                                )
                            
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                            
                            # Explicación del mapa de calor
                            st.markdown('<div class="explanation">', unsafe_allow_html=True)
                            st.markdown(f"""
                            **Interpretación del mapa de calor:**
                            - Colores más intensos indican mayor probabilidad de ser "{happiness_level}"
                            - La cruz blanca marca los valores medianos de ambas variables
                            - Los contornos blancos conectan puntos con la misma probabilidad
                            """)
                            
                            # Añadir interpretación específica según el tipo de interacción
                            if interaction_pval < 0.05:
                                if interaction_coef > 0:
                                    st.markdown(f"""
                                    La **interacción positiva significativa** indica que el efecto positivo de una variable 
                                    se intensifica cuando la otra variable aumenta. Esto se ve en la parte superior derecha del mapa 
                                    que muestra probabilidades más altas de lo que se esperaría si no hubiera interacción.
                                    """)
                                else:
                                    st.markdown(f"""
                                    La **interacción negativa significativa** indica que el efecto positivo de una variable 
                                    disminuye cuando la otra variable aumenta. Las probabilidades más altas tienden a concentrarse 
                                    cuando una variable es alta y la otra es baja.
                                    """)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        else:
                            # Para variables binarias, usar gráficos de líneas agrupadas
                            # Identificar cuál es la variable categórica
                            if var1 in ['female', 'victim']:
                                cat_var, num_var = var1, var2
                            else:
                                cat_var, num_var = var2, var1
                            
                            # Valores medios para variables no mostradas
                            default_values = {var: float(df[var].dropna().median()) for var in model_vars}
                            
                            # Crear valores para cada nivel de la variable categórica
                            cat_levels = [0, 1]
                            if cat_var == 'female':
                                cat_labels = gender_labels
                            elif cat_var == 'victim':
                                cat_labels = victim_labels
                            else:
                                cat_labels = ['Nivel 0', 'Nivel 1']  # Fallback
                            
                            # Crear valores para la variable numérica
                            num_values = np.linspace(df[num_var].dropna().min(), df[num_var].dropna().max(), 20)
                            
                            # Crear DataFrame para predicciones
                            pred_data = []
                            
                            for cat_val in cat_levels:
                                for num_val in num_values:
                                    temp_dict = default_values.copy()
                                    temp_dict[cat_var] = cat_val
                                    temp_dict[num_var] = num_val
                                    
                                    # Asegurarse de calcular la interacción correctamente
                                    temp_dict[interaction_name] = cat_val * num_val
                                    
                                    # Crear DataFrame para la predicción
                                    pred_df = pd.DataFrame([temp_dict])
                                    
                                    # Hacer predicción
                                    pred = interaction_mod.predict(interaction_res.params, exog=pred_df)
                                    
                                    # Guardar resultados para los cuatro niveles
                                    for i, level in enumerate(happiness_labels):
                                        pred_data.append({
                                            'cat_value': cat_val,
                                            'cat_label': cat_labels[cat_val],
                                            'num_value': num_val,
                                            'happiness_level': level,
                                            'probability': pred[0, i] * 100
                                        })
                            
                            # Convertir a DataFrame
                            pred_df = pd.DataFrame(pred_data)
                            
                            # Nivel de felicidad a mostrar
                            selected_level = st.radio(
                                "Nivel de felicidad a visualizar:",
                                happiness_labels,
                                index=3,  # Muy Feliz por defecto
                                horizontal=True,
                                key="line_happiness_level"
                            )
                            
                            # Filtrar por nivel de felicidad seleccionado
                            filtered_df = pred_df[pred_df['happiness_level'] == selected_level]
                            
                            # Crear gráfico de líneas mejorado
                            fig_lines = px.line(
                                filtered_df, 
                                x='num_value', 
                                y='probability',
                                color='cat_label',
                                labels={
                                    'num_value': var_labels.get(num_var, num_var),
                                    'probability': f'Probabilidad de {selected_level} (%)',
                                    'cat_label': var_labels.get(cat_var, cat_var)
                                },
                                title=f'Interacción entre {var_labels.get(cat_var, cat_var)} y {var_labels.get(num_var, num_var)}',
                                color_discrete_map={
                                    cat_labels[0]: '#3498db',
                                    cat_labels[1]: '#e74c3c'
                                }
                            )
                            
                            # Añadir marcadores a las líneas
                            fig_lines.update_traces(
                                mode='lines+markers', 
                                marker=dict(size=6),
                                hovertemplate=(
                                    f"{var_labels.get(num_var, num_var)}: %{{x:.2f}}<br>" +
                                    f"Probabilidad: %{{y:.1f}}%<extra>%{{fullData.name}}</extra>"
                                )
                            )
                            
                            # Mejorar layout
                            fig_lines.update_layout(
                                height=400,
                                margin=dict(l=0, r=0, t=50, b=30),
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                            )
                            
                            # Determinar si hay interacción significativa
                            if interaction_pval < 0.05:
                                # Añadir anotación señalando la interacción
                                if interaction_coef > 0:
                                    annotation_text = "Interacción positiva significativa"
                                    annotation_color = "rgba(46, 204, 113, 0.8)"
                                else:
                                    annotation_text = "Interacción negativa significativa"
                                    annotation_color = "rgba(231, 76, 60, 0.8)"
                                
                                fig_lines.add_annotation(
                                    xref="paper", yref="paper",
                                    x=0.5, y=1.07,
                                    text=annotation_text,
                                    showarrow=False,
                                    font=dict(color="white", size=12),
                                    bgcolor=annotation_color,
                                    bordercolor="rgba(0,0,0,0)",
                                    borderwidth=0,
                                    borderpad=4,
                                    width=220,
                                    align="center",
                                    # borderradius=5
                                )
                            
                            st.plotly_chart(fig_lines, use_container_width=True)
                            
                            # Explicación del gráfico
                            st.markdown('<div class="explanation">', unsafe_allow_html=True)
                            st.markdown(f"""
                            **Interpretación del gráfico de líneas:**
                            - Cada línea muestra cómo cambia la probabilidad de ser "{selected_level}" cuando varía {var_labels.get(num_var, num_var)}, para cada grupo de {var_labels.get(cat_var, cat_var)}
                            - Líneas paralelas indicarían que no hay interacción
                            - Líneas que convergen o divergen indican una interacción
                            """)
                            
                            # Añadir interpretación específica según el tipo de interacción
                            if interaction_pval < 0.05:
                                if interaction_coef > 0:
                                    st.markdown(f"""
                                    La **interacción positiva significativa** indica que el efecto de {var_labels.get(num_var, num_var)} 
                                    sobre la felicidad es más fuerte en el grupo "{cat_labels[1]}" comparado con el grupo "{cat_labels[0]}".
                                    """)
                                else:
                                    st.markdown(f"""
                                    La **interacción negativa significativa** indica que el efecto de {var_labels.get(num_var, num_var)} 
                                    sobre la felicidad es menos fuerte (o incluso opuesto) en el grupo "{cat_labels[1]}" comparado con el grupo "{cat_labels[0]}".
                                    """)
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No se pudieron obtener resultados completos para las variables seleccionadas.")
                
                with col_side:
                    # Detalles de la interacción
                    st.markdown("### 📊 Detalles de la Interacción")
                    
                    # Verificar que la interacción existe en los resultados
                    if interaction_name in interaction_results['Variable'].values:
                        # Información sobre la interacción
                        interaction_coef = interaction_results.loc[interaction_results['Variable'] == interaction_name, 'Coeficiente'].values[0]
                        interaction_pval = interaction_results.loc[interaction_results['Variable'] == interaction_name, 'p-valor'].values[0]
                        interaction_sig = interaction_pval < 0.05
                        
                        # Mostrar coeficiente con estilo según signo y significancia
                        coef_color = '#2ecc71' if interaction_coef > 0 and interaction_sig else '#e74c3c' if interaction_coef < 0 and interaction_sig else '#7f8c8d'
                        
                        st.markdown(f"""
                        <div style="
                            background-color: #f9f9f9;
                            border-radius: 10px;
                            padding: 15px;
                            margin-bottom: 15px;
                        ">
                            <h4 style="margin-top: 0;">Coeficiente de Interacción</h4>
                            <h2 style="color: {coef_color};">{interaction_coef:.3f}</h2>
                            <p>p-valor: {interaction_pval:.4f} <span style="font-weight: bold; color: {'#2ecc71' if interaction_sig else '#e74c3c'};">{'Significativo' if interaction_sig else 'No significativo'}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Interpretación de la interacción
                        st.markdown("### 🧩 Interpretación")
                        
                        var1_label = var_labels.get(var1, var1)
                        var2_label = var_labels.get(var2, var2)
                        
                        if interaction_sig:
                            if interaction_coef > 0:
                                st.markdown(f"""
                                <div style="
                                    background-color: rgba(46, 204, 113, 0.1);
                                    border-radius: 10px;
                                    padding: 15px;
                                    margin-bottom: 15px;
                                    border-left: 5px solid #2ecc71;
                                ">
                                    <p><b>Interacción positiva significativa</b></p>
                                    <p>El efecto de <b>{var1_label}</b> sobre la felicidad se intensifica a medida que aumenta <b>{var2_label}</b>.</p>
                                    <p>De manera similar, el efecto de <b>{var2_label}</b> es más fuerte cuando <b>{var1_label}</b> es mayor.</p>
                                    <p>Estas variables se <b>refuerzan mutuamente</b> en su impacto sobre la felicidad.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="
                                    background-color: rgba(231, 76, 60, 0.1);
                                    border-radius: 10px;
                                    padding: 15px;
                                    margin-bottom: 15px;
                                    border-left: 5px solid #e74c3c;
                                ">
                                    <p><b>Interacción negativa significativa</b></p>
                                    <p>El efecto positivo de <b>{var1_label}</b> sobre la felicidad disminuye a medida que aumenta <b>{var2_label}</b>.</p>
                                    <p>De manera similar, el efecto de <b>{var2_label}</b> es menos fuerte (o incluso opuesto) cuando <b>{var1_label}</b> es mayor.</p>
                                    <p>Estas variables <b>atenúan mutuamente</b> su impacto sobre la felicidad.</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="
                                background-color: rgba(127, 140, 141, 0.1);
                                border-radius: 10px;
                                padding: 15px;
                                margin-bottom: 15px;
                                border-left: 5px solid #7f8c8d;
                            ">
                                <p><b>Interacción no significativa</b></p>
                                <p>No hay evidencia suficiente para afirmar que exista una interacción entre <b>{var1_label}</b> y <b>{var2_label}</b>.</p>
                                <p>Estas variables influyen en la felicidad de manera <b>independiente</b>, sin reforzarse ni atenuarse mutuamente.</p>
                                <p>Los efectos de cada variable sobre la felicidad parecen ser aditivos.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Comparación con el modelo original
                        st.markdown("### 📈 Comparación de Modelos")
                        
                        # Log-likelihood del modelo original y con interacción
                        ll_original = model.llf
                        ll_interaction = interaction_res.llf
                        
                        # Test de razón de verosimilitud
                        lr_test = -2 * (ll_original - ll_interaction)
                        p_value = 1 - chi2.cdf(lr_test, df=1)
                        
                        # Crear métricas para comparación
                        metrics_table = pd.DataFrame({
                            'Métrica': ['Log-Likelihood', 'AIC', 'BIC', 'LR Test', 'p-valor'],
                            'Modelo Original': [f"{ll_original:.2f}", f"{model.aic:.2f}", f"{model.bic:.2f}", "-", "-"],
                            'Modelo con Interacción': [f"{ll_interaction:.2f}", f"{interaction_res.aic:.2f}", f"{interaction_res.bic:.2f}", f"{lr_test:.2f}", f"{p_value:.4f}"]
                        })
                        
                        st.dataframe(metrics_table, hide_index=True, use_container_width=True)
                        
                        # Mejora del modelo
                        if p_value < 0.05:
                            improvement_pct = (1 - np.exp(ll_original - ll_interaction)) * 100
                            
                            st.markdown(f"""
                            <div style="
                                background-color: rgba(46, 204, 113, 0.1);
                                border-radius: 10px;
                                padding: 15px;
                                margin-bottom: 15px;
                                border-left: 5px solid #2ecc71;
                            ">
                                <p><b>Mejora significativa del modelo (p = {p_value:.4f})</b></p>
                                <p>El modelo que incluye la interacción entre <b>{var1_label}</b> y <b>{var2_label}</b> es significativamente mejor que el modelo original.</p>
                                <p>La inclusión de esta interacción mejora el ajuste del modelo en aproximadamente un <b>{improvement_pct:.2f}%</b>.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="
                                background-color: rgba(127, 140, 141, 0.1);
                                border-radius: 10px;
                                padding: 15px;
                                margin-bottom: 15px;
                                border-left: 5px solid #7f8c8d;
                            ">
                                <p><b>No hay mejora significativa del modelo (p = {p_value:.4f})</b></p>
                                <p>No hay evidencia suficiente para afirmar que el modelo con interacción sea mejor que el modelo original.</p>
                                <p>La inclusión de esta interacción no mejora sustancialmente el ajuste del modelo.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"No se encontró el término de interacción {interaction_name} en los resultados del modelo.")
                
                # Tabla comparativa de coeficientes
                with st.expander("📋 Ver Comparación Completa de Coeficientes"):
                    # Verificar que tenemos ambos conjuntos de resultados
                    if 'model_results' in locals() and 'interaction_results' in locals():
                        try:
                            # Unir resultados
                            comparison = pd.merge(
                                model_results[['Variable', 'Coeficiente', 'p-valor']],
                                interaction_results[['Variable', 'Coeficiente', 'p-valor']],
                                on='Variable',
                                suffixes=(' (Original)', ' (Interacción)')
                            )
                            
                            # Añadir etiquetas de variables
                            comparison['Etiqueta'] = comparison['Variable'].apply(lambda x: var_labels.get(x, x))
                            
                            # Añadir columna de cambio porcentual
                            comparison['Cambio (%)'] = ((comparison['Coeficiente (Interacción)'] / comparison['Coeficiente (Original)']) - 1) * 100
                            
                            # Formatear para mostrar
                            comparison_display = comparison.copy()
                            comparison_display['Coeficiente (Original)'] = comparison_display['Coeficiente (Original)'].round(3)
                            comparison_display['Coeficiente (Interacción)'] = comparison_display['Coeficiente (Interacción)'].round(3)
                            comparison_display['p-valor (Original)'] = comparison_display['p-valor (Original)'].round(4)
                            comparison_display['p-valor (Interacción)'] = comparison_display['p-valor (Interacción)'].round(4)
                            comparison_display['Cambio (%)'] = comparison_display['Cambio (%)'].round(1)
                            
                            # Reordenar columnas
                            comparison_display = comparison_display[['Variable', 'Etiqueta', 'Coeficiente (Original)', 'p-valor (Original)', 'Coeficiente (Interacción)', 'p-valor (Interacción)', 'Cambio (%)']]
                            
                            # Destacar la interacción
                            if interaction_name in comparison_display['Variable'].values:
                                comparison_display.loc[comparison_display['Variable'] == interaction_name, 'Etiqueta'] = f"Interacción ({var_labels.get(var1, var1)} × {var_labels.get(var2, var2)})"
                            
                            # Mostrar tabla estilizada
                            st.dataframe(
                                comparison_display,
                                column_config={
                                    "Variable": "Código",
                                    "Etiqueta": "Variable",
                                    "Coeficiente (Original)": st.column_config.NumberColumn("Coef. Original", format="%.3f"),
                                    "p-valor (Original)": st.column_config.NumberColumn("p-valor Original", format="%.4f"),
                                    "Coeficiente (Interacción)": st.column_config.NumberColumn("Coef. Interacción", format="%.3f"),
                                    "p-valor (Interacción)": st.column_config.NumberColumn("p-valor Interacción", format="%.4f"),
                                    "Cambio (%)": st.column_config.NumberColumn("Cambio %", format="%.1f%%")
                                },
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error al crear la tabla comparativa: {str(e)}")
                    else:
                        st.warning("No se pudieron obtener los resultados de ambos modelos para hacer la comparación.")
            else:
                st.warning(f"No hay suficientes datos para ajustar el modelo de interacción. Se necesitan al menos {len(model_vars) + 10} observaciones completas.")
        except Exception as e:
            st.error(f"Error al ajustar el modelo de interacción: {str(e)}")
            st.info("Esto puede deberse a problemas de convergencia o multicolinealidad en los datos. Pruebe con variables diferentes o con más datos.")
    except NameError:
        st.warning("No hay un modelo ajustado disponible. Primero debe ir a la pestaña 'Modelo' para ajustar el modelo.")
        
        # Mostrar una explicación de las interacciones
        st.markdown("""
        ### 🛠️ Análisis de interacciones no disponible
        
        Para explorar interacciones entre variables, primero debe ir a la pestaña 'Modelo' y esperar a que se ajuste el modelo estadístico.
        
        Una vez que el modelo esté disponible, podrá:
        - Seleccionar dos variables para analizar su interacción
        - Visualizar cómo interactúan mediante mapas de calor o gráficos de líneas
        - Comparar el modelo con y sin interacción
        - Interpretar el significado de la interacción
        """)
        
        # Mostrar un ejemplo ilustrativo
        st.markdown("### 💡 Ejemplo de una interacción")
        
        st.markdown("""
        Una interacción significativa entre "Satisfacción Financiera" y "Salud" podría indicar que:
        
        - El efecto positivo de la satisfacción financiera en la felicidad es más fuerte en personas con mala salud
        - O que el efecto negativo de una mala salud es menor en personas con alta satisfacción financiera
        
        Esto sugeriría que una buena situación financiera podría compensar parcialmente los efectos negativos de problemas de salud.
        """)

# TAB 6: INSIGHTS CLAVE
with tab6:
    colored_header(
        label="Insights Clave sobre la Felicidad",
        description="Hallazgos principales y recomendaciones basadas en el análisis",
        color_name="blue-70"
    )
    
    st.markdown("""
    <div class="info-box">
    <h4>🧠 ¿Por qué es importante entender la felicidad?</h4>
    <p>La felicidad es un componente esencial del bienestar humano y está asociada con numerosos resultados positivos:</p>
    <ul>
        <li>Mejor salud física y mental</li>
        <li>Mayor longevidad y resistencia a enfermedades</li>
        <li>Mejores relaciones interpersonales y sociabilidad</li>
        <li>Mayor productividad y creatividad</li>
        <li>Decisiones más racionales y menor impulsividad</li>
    </ul>
    <p>Entender los factores que impulsan la felicidad puede ayudar a diseñar políticas públicas, intervenciones clínicas y estrategias personales más efectivas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Intentar acceder al modelo para ver si hay resultados disponibles
    have_model = False
    
    try:
        # Verificar si el modelo está definido y tiene resultados
        if 'model' in locals() and hasattr(model, 'model') and hasattr(model, 'params'):
            have_model = True
            
            # Top factores que afectan la felicidad (basados en magnitud de coeficientes)
            significant_vars = model_results[model_results['Significativo']]
            significant_vars['abs_coef'] = significant_vars['Coeficiente'].abs()
            top_factors = significant_vars.sort_values('abs_coef', ascending=False).head(5)
            
            # Preparar datos para la visualización
            top_factors['Etiqueta'] = top_factors['Variable'].apply(lambda x: var_labels.get(x, x))
            top_factors['Dirección'] = top_factors['Coeficiente'].apply(lambda x: 'Positivo' if x > 0 else 'Negativo')
            top_factors['Color'] = top_factors['Coeficiente'].apply(lambda x: '#2ecc71' if x > 0 else '#e74c3c')
        # Gráfico de factores principales
        st.subheader("📊 Principales Determinantes de la Felicidad")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de barras horizontales
            fig_top = px.bar(
                top_factors,
                y='Etiqueta',
                x='abs_coef',
                color='Dirección',
                text='Coeficiente',
                labels={
                    'abs_coef': 'Magnitud del Efecto',
                    'Etiqueta': 'Factor'
                },
                title='Top 5 Factores que Afectan la Felicidad',
                color_discrete_map={
                    'Positivo': '#2ecc71',
                    'Negativo': '#e74c3c'
                },
                height=400
            )
            
            fig_top.update_traces(
                texttemplate='%{text:.3f}',
                textposition='outside'
            )
            
            fig_top.update_layout(
                xaxis_title='Magnitud del Efecto (|Coeficiente|)',
                yaxis=dict(categoryorder='total ascending')
            )
            
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            # Resumen de los principales factores
            st.markdown("### Factores Clave")
            
            for i, (_, row) in enumerate(top_factors.iterrows()):
                effect = "aumenta" if row['Coeficiente'] > 0 else "disminuye"
                emoji = "⬆️" if row['Coeficiente'] > 0 else "⬇️"
                
                st.markdown(f"""
                <div style="
                    background-color: {row['Color']}10;
                    border-radius: 8px;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-left: 4px solid {row['Color']};
                ">
                    <p style="margin: 0;"><b>{i+1}. {row['Etiqueta']}</b> {emoji}</p>
                    <p style="margin: 0; font-size: 0.9rem;">Mayor {row['Etiqueta'].lower()} {effect} la felicidad.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Insights principales basados en los resultados del modelo
        st.markdown("### 💡 Insights Principales")
        
        # Contenedor de tarjetas
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        
        # Determinar las categorías principales basadas en los resultados
        has_social = any(var in significant_vars['Variable'].values for var in ['rel_import', 'imp_family', 'imp_friends'])
        has_economic = any(var in significant_vars['Variable'].values for var in ['fin_sat'])
        has_health = any(var in significant_vars['Variable'].values for var in ['health'])
        has_security = any(var in significant_vars['Variable'].values for var in ['sec_rec', 'victim', 'indice_criminalidad'])
        has_control = any(var in significant_vars['Variable'].values for var in ['control'])
        has_gender = 'female' in significant_vars['Variable'].values
        
        # Insight 1: Satisfacción con la vida
        if 'life_sat' in significant_vars['Variable'].values:
            life_sat_coef = significant_vars.loc[significant_vars['Variable'] == 'life_sat', 'Coeficiente'].values[0]
            create_insight_card(
                "La satisfacción general afecta a la felicidad",
                f"La satisfacción general con la vida tiene un efecto {'positivo' if life_sat_coef > 0 else 'negativo'} y significativo en la felicidad. " + 
                "Esto sugiere que la evaluación cognitiva de la propia vida está fuertemente conectada con la experiencia emocional de la felicidad.",
                "🌟"
            )
        
        # Insight 2: Salud
        if has_health:
            health_coef = significant_vars.loc[significant_vars['Variable'] == 'health', 'Coeficiente'].values[0]
            create_insight_card(
                "La salud es fundamental para la felicidad",
                f"El estado de salud autopercibido tiene un efecto {'positivo' if health_coef > 0 else 'negativo'} y significativo en la felicidad. " + 
                "Esto resalta la importancia de mantener una buena salud física y mental como base para el bienestar emocional.",
                "💪"
            )
        
        # Insight 3: Factores económicos
        if has_economic:
            econ_coef = significant_vars.loc[significant_vars['Variable'] == 'fin_sat', 'Coeficiente'].values[0]
            create_insight_card(
                "El dinero SÍ compra la felicidad (hasta cierto punto)",
                f"La satisfacción financiera tiene un efecto {'positivo' if econ_coef > 0 else 'negativo'} significativo en la felicidad. " + 
                "Esto confirma que la seguridad económica y la satisfacción con los recursos disponibles son importantes para el bienestar subjetivo.",
                "💰"
            )
        
        # Insight 4: Relaciones sociales
        if has_social:
            social_var = next((var for var in ['rel_import', 'imp_family', 'imp_friends'] if var in significant_vars['Variable'].values), None)
            if social_var:
                social_coef = significant_vars.loc[significant_vars['Variable'] == social_var, 'Coeficiente'].values[0]
                create_insight_card(
                    "Las relaciones sociales predicen la felicidad",
                    f"La importancia dada a las relaciones sociales tiene un efecto {'positivo' if social_coef > 0 else 'negativo'} significativo en la felicidad. " + 
                    "Este hallazgo coincide con décadas de investigación que muestran que las conexiones sociales son uno de los predictores más consistentes del bienestar.",
                    "👪"
                )
        
        # Insight 5: Seguridad
        if has_security:
            security_var = next((var for var in ['sec_rec', 'indice_criminalidad'] if var in significant_vars['Variable'].values), None)
            if security_var:
                sec_coef = significant_vars.loc[significant_vars['Variable'] == security_var, 'Coeficiente'].values[0]
                direction = "positivo" if sec_coef > 0 else "negativo"
                if security_var == 'indice_criminalidad':
                    direction = "negativo" if sec_coef > 0 else "positivo"
                
                create_insight_card(
                    "La seguridad es clave para la felicidad",
                    f"Los indicadores de seguridad tienen un efecto {direction} significativo en la felicidad. " + 
                    "La percepción de vivir en un entorno seguro permite que las personas se sientan más tranquilas y libres para disfrutar de la vida.",
                    "🔒"
                )
        
        # Insight 6: Género
        if has_gender:
            gender_coef = significant_vars.loc[significant_vars['Variable'] == 'female', 'Coeficiente'].values[0]
            create_insight_card(
                "Hay diferencias de género en la felicidad",
                f"Ser mujer tiene un efecto {'positivo' if gender_coef > 0 else 'negativo'} significativo en la felicidad. " + 
                "Esta diferencia puede reflejar patrones culturales, roles sociales u otros factores que afectan de manera diferenciada a hombres y mujeres.",
                "⚤"
            )
        
        # Insight 7: Control
        if has_control:
            control_coef = significant_vars.loc[significant_vars['Variable'] == 'control', 'Coeficiente'].values[0]
            create_insight_card(
                "El control percibido influye en la felicidad",
                f"La percepción de control sobre la propia vida tiene un efecto {'positivo' if control_coef > 0 else 'negativo'} significativo en la felicidad. " + 
                "Las personas que sienten que pueden influir en su destino tienden a experimentar mayor bienestar emocional.",
                "🎮"
            )
            
        # Cerrar contenedor de tarjetas
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recomendaciones basadas en los resultados
        st.markdown("### 📝 Recomendaciones")
        
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        
        st.markdown("""
        Basándose en los resultados del análisis, se pueden derivar las siguientes recomendaciones para aumentar los niveles de felicidad:
        
        #### A nivel personal:
        
        - **Cuidar la salud física y mental** como prioridad fundamental
        - **Cultivar relaciones sociales significativas** con familia y amigos
        - **Buscar equilibrio financiero** y satisfacción con los recursos disponibles
        - **Desarrollar un sentido de propósito y control** sobre la propia vida
        - **Crear entornos seguros** y evitar situaciones de riesgo
        
        #### A nivel de políticas públicas:
        
        - **Promover el acceso universal a servicios de salud** de calidad
        - **Fomentar políticas de seguridad** que reduzcan la criminalidad y aumenten la percepción de seguridad
        - **Implementar programas de bienestar financiero** y reducción de la desigualdad
        - **Crear espacios públicos que faciliten la interacción social** y el sentido de comunidad
        - **Desarrollar intervenciones específicas** para grupos con menores niveles de felicidad
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Limitaciones del estudio
        with st.expander("⚠️ Limitaciones del Análisis"):
            st.markdown("""
            Al interpretar estos resultados, es importante tener en cuenta las siguientes limitaciones:
            
            - **Causalidad**: Este análisis muestra asociaciones, pero no prueba relaciones causales directas
            - **Auto-reporte**: Los datos provienen de encuestas de auto-reporte, que pueden estar sujetas a sesgos
            - **Diferencias culturales**: La definición y expresión de la felicidad puede variar entre culturas
            - **Variables omitidas**: Pueden existir factores importantes que no se incluyeron en el análisis
            - **Generalización**: Los resultados pueden no aplicarse por igual a todas las poblaciones y contextos
            
            A pesar de estas limitaciones, los patrones identificados son consistentes con la literatura existente sobre bienestar subjetivo y pueden ofrecer orientación valiosa.
            """)
    
    except Exception as e:
        # Si no hay modelo, mostrar insights generales basados en la literatura
        st.subheader("💡 Insights Generales sobre la Felicidad")
        
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        
        # Insight 1
        create_insight_card(
            "La salud es un pilar fundamental",
            "Numerosos estudios muestran que el estado de salud física y mental es uno de los predictores más fuertes de la felicidad. Las personas con buena salud tienden a reportar niveles significativamente más altos de bienestar subjetivo.",
            "💪"
        )
        
        # Insight 2
        create_insight_card(
            "Las relaciones sociales son cruciales",
            "La calidad de las relaciones sociales, especialmente con familiares y amigos cercanos, es consistentemente uno de los predictores más potentes de la felicidad. El apoyo social actúa como un amortiguador contra el estrés y las dificultades de la vida.",
            "👪"
        )
        
        # Insight 3
        create_insight_card(
            "El dinero importa, pero con rendimientos decrecientes",
            "Los recursos económicos tienen un impacto significativo en la felicidad, pero este efecto tiende a disminuir después de cubrir las necesidades básicas. La satisfacción financiera subjetiva suele ser más importante que el ingreso absoluto.",
            "💰"
        )
        
        # Insight 4
        create_insight_card(
            "El sentido de propósito es esencial",
            "Las personas que tienen un sentido claro de propósito y significado en sus vidas tienden a reportar mayor felicidad. Esto puede provenir de la religión, la familia, el trabajo o el voluntariado, entre otras fuentes.",
            "🎯"
        )
        
        # Insight 5
        create_insight_card(
            "La seguridad es un prerrequisito",
            "Vivir en un entorno seguro y estable es una condición fundamental para la felicidad. La preocupación constante por la seguridad personal o familiar agota los recursos psicológicos y reduce significativamente el bienestar.",
            "🔒"
        )
        
        # Insight 6
        create_insight_card(
            "La adaptación hedónica modula la felicidad",
            "Los seres humanos tienen una tendencia a adaptarse a las nuevas circunstancias, tanto positivas como negativas. Esta 'adaptación hedónica' explica por qué los grandes cambios vitales suelen tener un impacto temporal en la felicidad.",
            "🔄"
        )
        
        # Cerrar contenedor de tarjetas
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.info("Estos insights se basan en la literatura científica sobre felicidad y bienestar subjetivo. Para un análisis específico basado en este conjunto de datos, vaya a la pestaña 'Modelo'.")
    
    # Conclusión general
    st.subheader("🎯 Conclusión")
    
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    
    if have_model:
        # Conclusión basada en los resultados del modelo
        st.markdown("""
        El análisis de los determinantes de la felicidad muestra que esta es un fenómeno multidimensional influido por una variedad de factores:
        
        - La **satisfacción con la vida** y la **salud** emergen como pilares fundamentales del bienestar subjetivo
        - Las **relaciones sociales significativas** aportan un valor esencial a la experiencia humana
        - La **seguridad financiera** y la **sensación de control** sobre la propia vida contribuyen notablemente
        - El **entorno seguro** y libre de amenazas proporciona la base para poder disfrutar de otros aspectos de la vida
        
        La felicidad no depende de un único factor dominante, sino de un equilibrio entre diversos aspectos de la vida. Esta visión holística sugiere que las intervenciones para mejorar el bienestar deberían ser multifacéticas, abordando tanto necesidades básicas como aspectos psicológicos y sociales más complejos.
        """)
    else:
        # Conclusión general sin modelo específico
        st.markdown("""
        La investigación científica sobre la felicidad revela consistentemente su naturaleza multifacética:
        
        - La felicidad depende de una combinación de factores que incluyen salud, relaciones sociales, seguridad financiera y propósito vital
        - Existe un equilibrio entre factores "externos" (circunstancias) e "internos" (actitudes, valores y percepciones)
        - Las intervenciones más efectivas para aumentar la felicidad tienden a ser aquellas que abordan múltiples dimensiones del bienestar
        - La capacidad de adaptación humana sugiere que fomentar hábitos y perspectivas positivas puede ser más efectivo que cambios circunstanciales puntuales
        
        Comprender estos patrones puede ayudar tanto a individuos como a sociedades a desarrollar estrategias más efectivas para promover el bienestar generalizado.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recursos adicionales
    st.subheader("📚 Recursos Adicionales")
    
    st.markdown("""
    Para profundizar en el estudio de la felicidad y el bienestar subjetivo:
    
    - **World Happiness Report**: [https://worldhappiness.report/](https://worldhappiness.report/)
    - **World Values Survey**: [https://www.worldvaluessurvey.org/](https://www.worldvaluessurvey.org/)
    - **Positive Psychology Center**: [https://ppc.sas.upenn.edu/](https://ppc.sas.upenn.edu/)
    - **Journal of Happiness Studies**: [https://www.springer.com/journal/10902](https://www.springer.com/journal/10902)
    - **OECD Better Life Index**: [https://www.oecdbetterlifeindex.org/](https://www.oecdbetterlifeindex.org/)
    """)

# PIE DE PÁGINA
st.markdown("""---""")
st.markdown("""
<div style="text-align: center">
    <p>Dashboard Premium para el Análisis de Determinantes de la Felicidad</p>
    <p>Basado en datos del World Values Survey</p>
    <p>Dashboard creado por Eduardo Alzu, Diego Antón, Marcos Domínguez, Roberto García, Ignacio Fumanal y Ángel Prados</p>
</div>
""", unsafe_allow_html=True)
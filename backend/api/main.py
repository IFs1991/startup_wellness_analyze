from api.routers import (
    auth,
    users,
    health,
    wellness_scores,
    dashboard,
    recommendations,
    goals,
    correlation_visualization,  # 相関分析可視化
    timeseries_analysis_visualization,  # 時系列分析可視化
    statistical_analysis_visualization,  # 統計分析可視化
)

# APIルーターの登録
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(health.router)
app.include_router(wellness_scores.router)
app.include_router(dashboard.router)
app.include_router(recommendations.router)
app.include_router(goals.router)
app.include_router(correlation_visualization.router)  # 相関分析可視化ルーター
app.include_router(timeseries_analysis_visualization.router)  # 時系列分析可視化ルーター
app.include_router(statistical_analysis_visualization.router)  # 統計分析可視化ルーター
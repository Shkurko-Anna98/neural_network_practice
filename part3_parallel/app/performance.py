"""
Модуль для сохранения результатов замеров времени.
"""
import json

def save_performance_report(sequential_time, parallel_time, filename="performance_report.json"):
    """Записывает время работы в JSON файл."""
    report = {
        "sequential_time_sec": round(sequential_time, 4),
        "parallel_time_sec": round(parallel_time, 4),
        "speedup": round(sequential_time / parallel_time, 2) if parallel_time > 0 else 0
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"\nОтчет сохранен в файл: {filename}")
    print(f"Последовательно: {report['sequential_time_sec']} сек.")
    print(f"Параллельно: {report['parallel_time_sec']} сек.")
    print(f"Ускорение: x{report['speedup']}")
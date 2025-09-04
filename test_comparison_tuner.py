"""
測試比對調優功能的簡單示例
"""

import json
import tempfile
from pathlib import Path

def test_comparison_tuner():
    """測試ComparisonTuner基本功能"""
    print("🧪 開始測試ComparisonTuner功能...")
    
    try:
        from hypersurrogatemodel import TrainableLLM, ComparisonTuner
        print("✅ 模組導入成功")
        
        # 創建測試數據集
        test_data = [
            {
                "text": "選擇適合空氣動力學模擬的代理模型",
                "answer": "神經網路"
            },
            {
                "text": "簡單線性關係建模",
                "answer": "線性回歸"
            }
        ]
        
        # 創建臨時數據集文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            temp_dataset_path = f.name
        
        print(f"✅ 測試數據集創建成功: {temp_dataset_path}")
        
        # 初始化模型（使用小模型以節省資源）
        print("🤖 初始化模型...")
        model = TrainableLLM(
            base_model_name="google/gemma-3-270m-it",
            use_lora=True,
        )
        
        print("✅ 模型初始化成功")
        
        # 初始化ComparisonTuner
        print("🔧 初始化ComparisonTuner...")
        tuner = ComparisonTuner(
            model=model,
            tokenizer=model.get_tokenizer(),
            output_dir="./test_results",
            use_wandb=False
        )
        
        print("✅ ComparisonTuner初始化成功")
        
        # 測試數據集載入和比對
        print("📊 測試數據集載入和比對...")
        try:
            comparison_results = tuner.load_and_compare_dataset(
                dataset_path=temp_dataset_path,
                text_column="text",
                answer_column="answer", 
                task_type="generation",
                comparison_method="similarity"
            )
            
            print("✅ 數據集比對成功")
            print(f"   總樣本數: {comparison_results['overall_metrics']['total_samples']}")
            print(f"   準確率: {comparison_results['overall_metrics']['accuracy']:.3f}")
            print(f"   平均相似度: {comparison_results['overall_metrics']['average_similarity']:.3f}")
            
        except Exception as e:
            print(f"❌ 數據集比對失敗: {e}")
            return False
        
        # 清理臨時文件
        Path(temp_dataset_path).unlink()
        print("🧹 清理臨時文件完成")
        
        print("🎉 所有測試通過！ComparisonTuner功能正常")
        return True
        
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

if __name__ == "__main__":
    success = test_comparison_tuner()
    if success:
        print("\\n✨ ComparisonTuner功能測試成功！")
        print("\\n您現在可以使用以下功能：")
        print("1. 載入數據集並與LLM預測比對")
        print("2. 分析預測差異和錯誤模式") 
        print("3. 基於差異進行自適應調優")
        print("4. 追蹤改進效果")
        print("\\n查看 examples/quick_start_comparison.py 獲取使用示例")
    else:
        print("\\n❌ 測試失敗，請檢查錯誤訊息")

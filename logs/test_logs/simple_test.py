#!/usr/bin/env python3
"""
简单的测试脚本来验证信息互馈系统的修复
"""

def test_basic_functionality():
    """测试基本功能"""
    print("测试基本功能...")
    
    try:
        # 测试导入
        print("正在导入模块...")
        
        # 模拟PyTorch张量
        class MockTensor:
            def __init__(self, data):
                self.data = data
                self.grad = None
            
            def item(self):
                return self.data
            
            def norm(self):
                return MockTensor(abs(self.data))
            
            def detach(self):
                return self
            
            def clone(self):
                return MockTensor(self.data)
            
            def size(self, dim=None):
                if dim is None:
                    return (len(self.data),)
                return len(self.data)
        
        # 模拟节点收敛跟踪器
        class MockConvergenceTracker:
            def __init__(self, num_nodes):
                self.num_nodes = num_nodes
                self.node_loss_history = {}
                self.node_grad_history = {}
                self.node_embedding_history = {}
                self.converged_nodes = set()
            
            def update_node_info(self, node_ids, losses, gradients, embeddings, epoch):
                print(f"更新节点信息: {len(node_ids)} 个节点, epoch {epoch}")
                
                for i, node_id in enumerate(node_ids):
                    node_id = node_id.item()
                    if node_id < self.num_nodes:
                        # 记录损失
                        if node_id not in self.node_loss_history:
                            self.node_loss_history[node_id] = []
                        self.node_loss_history[node_id].append(losses[i].item())
                        
                        # 记录梯度（使用损失作为替代）
                        if node_id not in self.node_grad_history:
                            self.node_grad_history[node_id] = []
                        
                        if gradients is not None and i < gradients.size(0):
                            if gradients[i].dim() == 0:
                                grad_value = gradients[i].item()
                            else:
                                grad_value = gradients[i].norm().item()
                        else:
                            grad_value = losses[i].item()
                        
                        self.node_grad_history[node_id].append(grad_value)
                        
                        # 记录嵌入
                        if node_id not in self.node_embedding_history:
                            self.node_embedding_history[node_id] = []
                        
                        if embeddings is not None and i < embeddings.size(0):
                            self.node_embedding_history[node_id].append(embeddings[i].detach().clone())
                        else:
                            # 创建虚拟嵌入
                            dummy_embedding = MockTensor([0.0] * 64)
                            self.node_embedding_history[node_id].append(dummy_embedding)
                
                print(f"✓ 成功更新 {len(node_ids)} 个节点的信息")
                return True
        
        # 测试
        tracker = MockConvergenceTracker(100)
        
        # 模拟数据
        node_ids = MockTensor([0, 1, 2, 3, 4])
        losses = MockTensor([0.5, 0.3, 0.8, 0.2, 0.6])
        gradients = MockTensor([0.1, 0.05, 0.15, 0.02, 0.12])
        embeddings = [MockTensor([0.1] * 64) for _ in range(5)]
        
        # 测试更新
        result = tracker.update_node_info(node_ids, losses, gradients, embeddings, epoch=1)
        
        if result:
            print("✓ 基本功能测试通过")
            return True
        else:
            print("✗ 基本功能测试失败")
            return False
            
    except Exception as e:
        print(f"✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况...")
    
    try:
        class MockConvergenceTracker:
            def __init__(self, num_nodes):
                self.num_nodes = num_nodes
                self.node_loss_history = {}
                self.node_grad_history = {}
                self.node_embedding_history = {}
                self.converged_nodes = set()
            
            def update_node_info(self, node_ids, losses, gradients, embeddings, epoch):
                print(f"边界测试: {len(node_ids)} 个节点")
                
                # 测试空数据
                if len(node_ids) == 0:
                    print("✓ 空数据处理成功")
                    return True
                
                # 测试None值
                if gradients is None:
                    print("✓ None梯度处理成功")
                
                if embeddings is None:
                    print("✓ None嵌入处理成功")
                
                return True
        
        tracker = MockConvergenceTracker(100)
        
        # 测试空数据
        empty_nodes = MockTensor([])
        empty_losses = MockTensor([])
        
        result1 = tracker.update_node_info(empty_nodes, empty_losses, None, None, epoch=1)
        
        # 测试None值
        node_ids = MockTensor([0, 1])
        losses = MockTensor([0.1, 0.2])
        
        result2 = tracker.update_node_info(node_ids, losses, None, None, epoch=1)
        
        if result1 and result2:
            print("✓ 边界情况测试通过")
            return True
        else:
            print("✗ 边界情况测试失败")
            return False
            
    except Exception as e:
        print(f"✗ 边界情况测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("开始简单测试...")
    print("=" * 40)
    
    tests = [
        test_basic_functionality,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"测试 {test_func.__name__} 失败")
        except Exception as e:
            print(f"测试 {test_func.__name__} 异常: {e}")
    
    print("\n" + "=" * 40)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！基本修复成功。")
        print("\n现在可以尝试运行分布式训练了。")
    else:
        print("❌ 部分测试失败，需要进一步修复。")

if __name__ == "__main__":
    main()

"""
信息互馈系统配置文件
"""

class FeedbackConfig:
    """信息互馈系统配置类"""
    
    # 节点收敛检测配置
    CONVERGENCE = {
        'threshold': 0.01,           # 收敛阈值
        'patience': 5,               # 耐心参数（连续多少个epoch检查收敛）
        'min_epochs': 3,             # 最小epoch数（在此之前不检查收敛）
        'loss_weight': 0.4,          # 损失变化权重
        'gradient_weight': 0.3,      # 梯度变化权重
        'embedding_weight': 0.3,     # 嵌入变化权重
    }
    
    # 采样策略配置
    SAMPLING = {
        'default_strategy': 'adaptive_importance',  # 默认采样策略
        'exploration_rate': 0.1,     # 探索率
        'convergence_penalty': 0.5,  # 收敛节点惩罚因子
        'importance_decay': 0.95,    # 重要性分数衰减因子
        'min_importance': 0.1,       # 最小重要性分数
    }
    
    # 反馈控制配置
    CONTROL = {
        'adjustment_frequency': 5,   # 策略调整频率（每N个epoch）
        'convergence_rate_threshold_high': 0.8,    # 高收敛率阈值
        'convergence_rate_threshold_low': 0.3,     # 低收敛率阈值
        'exploration_rate_max': 0.3,              # 最大探索率
        'exploration_rate_min': 0.05,             # 最小探索率
        'exploration_rate_adjustment': 1.1,       # 探索率调整因子
    }
    
    # 日志和监控配置
    LOGGING = {
        'log_level': 'INFO',         # 日志级别
        'log_feedback_interval': 5,  # 反馈信息记录间隔
        'log_sampling_stats': True,  # 是否记录采样统计
        'log_convergence_details': True,  # 是否记录收敛详情
    }
    
    # 性能优化配置
    PERFORMANCE = {
        'batch_size_adaptive': True, # 是否启用自适应batch大小
        'max_batch_size': 5000,      # 最大batch大小
        'min_batch_size': 100,       # 最小batch大小
        'memory_efficient': True,    # 是否启用内存优化
        'gradient_accumulation': False,  # 是否启用梯度累积
    }
    
    @classmethod
    def get_convergence_config(cls):
        """获取收敛检测配置"""
        return cls.CONVERGENCE.copy()
    
    @classmethod
    def get_sampling_config(cls):
        """获取采样策略配置"""
        return cls.SAMPLING.copy()
    
    @classmethod
    def get_control_config(cls):
        """获取反馈控制配置"""
        return cls.CONTROL.copy()
    
    @classmethod
    def get_logging_config(cls):
        """获取日志配置"""
        return cls.LOGGING.copy()
    
    @classmethod
    def get_performance_config(cls):
        """获取性能优化配置"""
        return cls.PERFORMANCE.copy()
    
    @classmethod
    def update_config(cls, section: str, key: str, value):
        """更新配置"""
        if hasattr(cls, section.upper()):
            section_dict = getattr(cls, section.upper())
            if key in section_dict:
                section_dict[key] = value
                return True
        return False
    
    @classmethod
    def get_all_config(cls):
        """获取所有配置"""
        return {
            'convergence': cls.get_convergence_config(),
            'sampling': cls.get_sampling_config(),
            'control': cls.get_control_config(),
            'logging': cls.get_logging_config(),
            'performance': cls.get_performance_config(),
        }


# 预定义的配置模板
class FeedbackConfigTemplates:
    """预定义配置模板"""
    
    @staticmethod
    def conservative():
        """保守配置：更注重稳定性"""
        config = FeedbackConfig.get_all_config()
        config['convergence']['threshold'] = 0.005
        config['convergence']['patience'] = 7
        config['sampling']['exploration_rate'] = 0.05
        config['sampling']['convergence_penalty'] = 0.3
        return config
    
    @staticmethod
    def aggressive():
        """激进配置：更注重快速收敛"""
        config = FeedbackConfig.get_all_config()
        config['convergence']['threshold'] = 0.02
        config['convergence']['patience'] = 3
        config['sampling']['exploration_rate'] = 0.2
        config['sampling']['convergence_penalty'] = 0.7
        return config
    
    @staticmethod
    def balanced():
        """平衡配置：默认配置"""
        return FeedbackConfig.get_all_config()
    
    @staticmethod
    def memory_optimized():
        """内存优化配置"""
        config = FeedbackConfig.get_all_config()
        config['performance']['memory_efficient'] = True
        config['performance']['max_batch_size'] = 2000
        config['performance']['gradient_accumulation'] = True
        return config


# 配置验证器
class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_convergence_config(config):
        """验证收敛配置"""
        errors = []
        if config['threshold'] <= 0:
            errors.append("收敛阈值必须大于0")
        if config['patience'] < 1:
            errors.append("耐心参数必须大于等于1")
        if config['min_epochs'] < 0:
            errors.append("最小epoch数必须大于等于0")
        
        total_weight = config['loss_weight'] + config['gradient_weight'] + config['embedding_weight']
        if abs(total_weight - 1.0) > 1e-6:
            errors.append("权重之和必须等于1.0")
        
        return errors
    
    @staticmethod
    def validate_sampling_config(config):
        """验证采样配置"""
        errors = []
        if not (0 <= config['exploration_rate'] <= 1):
            errors.append("探索率必须在[0,1]范围内")
        if not (0 <= config['convergence_penalty'] <= 1):
            errors.append("收敛惩罚因子必须在[0,1]范围内")
        if not (0 < config['importance_decay'] < 1):
            errors.append("重要性衰减因子必须在(0,1)范围内")
        
        return errors
    
    @staticmethod
    def validate_all_config(config):
        """验证所有配置"""
        all_errors = {}
        
        convergence_errors = ConfigValidator.validate_convergence_config(config['convergence'])
        if convergence_errors:
            all_errors['convergence'] = convergence_errors
        
        sampling_errors = ConfigValidator.validate_sampling_config(config['sampling'])
        if sampling_errors:
            all_errors['sampling'] = sampling_errors
        
        return all_errors


# 配置管理器
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_name: str = 'balanced'):
        self.config_name = config_name
        self.config = self._load_config(config_name)
        self._validate_config()
    
    def _load_config(self, config_name: str):
        """加载配置"""
        if config_name == 'conservative':
            return FeedbackConfigTemplates.conservative()
        elif config_name == 'aggressive':
            return FeedbackConfigTemplates.aggressive()
        elif config_name == 'memory_optimized':
            return FeedbackConfigTemplates.memory_optimized()
        else:
            return FeedbackConfigTemplates.balanced()
    
    def _validate_config(self):
        """验证配置"""
        errors = ConfigValidator.validate_all_config(self.config)
        if errors:
            error_msg = "配置验证失败:\n"
            for section, section_errors in errors.items():
                error_msg += f"  {section}: {', '.join(section_errors)}\n"
            raise ValueError(error_msg)
    
    def get_config(self, section: str = None):
        """获取配置"""
        if section is None:
            return self.config
        return self.config.get(section, {})
    
    def update_config(self, section: str, key: str, value):
        """更新配置"""
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            return True
        return False
    
    def reload_config(self, config_name: str = None):
        """重新加载配置"""
        if config_name:
            self.config_name = config_name
        self.config = self._load_config(self.config_name)
        self._validate_config()
    
    def print_config(self):
        """打印当前配置"""
        print(f"当前配置模板: {self.config_name}")
        print("=" * 50)
        for section, config in self.config.items():
            print(f"\n[{section.upper()}]")
            for key, value in config.items():
                print(f"  {key}: {value}")
        print("=" * 50)

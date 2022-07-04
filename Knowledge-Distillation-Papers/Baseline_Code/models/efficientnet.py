from efficientnet_pytorch import EfficientNet

def efficientnet_b0(pretrained=True, **kwargs):
    net_name = 'efficientnet-b0'
    model = EfficientNet.from_pretrained(net_name, num_classes=kwargs['num_classes']) if pretrained \
        else EfficientNet.from_name(net_name, num_classes=kwargs['num_classes'])
    return model

def efficientnet_b1(pretrained=True, **kwargs):
    net_name = 'efficientnet-b1'
    model = EfficientNet.from_pretrained(net_name, num_classes=kwargs['num_classes'], advprop=True) if pretrained \
        else EfficientNet.from_name(net_name, num_classes=kwargs['num_classes'], advprop=True)
    return model

def efficientnet_b2(pretrained=True, **kwargs):
    net_name = 'efficientnet-b2'
    model = EfficientNet.from_pretrained(net_name, num_classes=kwargs['num_classes'], advprop=True) if pretrained \
        else EfficientNet.from_name(net_name, num_classes=kwargs['num_classes'], advprop=True)
    return model

def efficientnet_b3(pretrained=True, **kwargs):
    net_name = 'efficientnet-b3'
    model = EfficientNet.from_pretrained(net_name, num_classes=kwargs['num_classes'], advprop=True) if pretrained \
        else EfficientNet.from_name(net_name, num_classes=kwargs['num_classes'], advprop=True)
    return model

def efficientnet_b4(pretrained=True, **kwargs):
    net_name = 'efficientnet-b4'
    model = EfficientNet.from_pretrained(net_name, num_classes=kwargs['num_classes'], advprop=True) if pretrained \
        else EfficientNet.from_name(net_name, num_classes=kwargs['num_classes'], advprop=True)
    return model

def efficientnet_b5(pretrained=True, **kwargs):
    net_name = 'efficientnet-b5'
    model = EfficientNet.from_pretrained(net_name, num_classes=kwargs['num_classes'], advprop=True) if pretrained \
        else EfficientNet.from_name(net_name, num_classes=kwargs['num_classes'], advprop=True)
    return model

def efficientnet_b6(pretrained=True, **kwargs):
    net_name = 'efficientnet-b6'
    model = EfficientNet.from_pretrained(net_name, num_classes=kwargs['num_classes'], advprop=True) if pretrained \
        else EfficientNet.from_name(net_name, num_classes=kwargs['num_classes'], advprop=True)
    return model

def efficientnet_b7(pretrained=True, **kwargs):
    net_name = 'efficientnet-b7'
    model = EfficientNet.from_pretrained(net_name, num_classes=kwargs['num_classes'], advprop=True) if pretrained \
        else EfficientNet.from_name(net_name, num_classes=kwargs['num_classes'], advprop=True)
    return model

import { requireNativeViewManager } from 'expo-modules-core';
import { ViewProps } from 'react-native';

const NativeView = requireNativeViewManager('BackgroundCamera');

export default function BackgroundCameraView(props: ViewProps) {
    return <NativeView {...props} />;
}

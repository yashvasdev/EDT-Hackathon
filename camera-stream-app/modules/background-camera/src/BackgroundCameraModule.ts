import { NativeModule, requireNativeModule } from 'expo-modules-core';

interface ConcurrentSupportResult {
    supported: boolean;
    reason: string;
    frontCameraId?: string;
    backCameraId?: string;
}

interface StartCaptureResult {
    started: boolean;
    reason: string;
}

interface FrameEvent {
    base64: string;
    timestamp: number;
}

interface ErrorEvent {
    message: string;
}

declare class BackgroundCameraModuleType extends NativeModule<{
    onFrame: (event: FrameEvent) => void;
    onError: (event: ErrorEvent) => void;
}> {
    checkConcurrentSupport(): Promise<ConcurrentSupportResult>;
    startCapture(intervalMs: number): Promise<StartCaptureResult>;
    stopCapture(): Promise<boolean>;
}

export default requireNativeModule<BackgroundCameraModuleType>('BackgroundCamera');

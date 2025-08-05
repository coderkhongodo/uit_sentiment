// Date utility functions to handle timestamps safely

export const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Vừa xong';

    try {
        const date = new Date(timestamp);
        if (isNaN(date.getTime())) return 'Vừa xong';

        return date.toLocaleString('vi-VN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (error) {
        return 'Vừa xong';
    }
};

export const formatTime = (timestamp) => {
    if (!timestamp) return 'Vừa xong';

    try {
        const date = new Date(timestamp);
        if (isNaN(date.getTime())) return 'Vừa xong';

        return date.toLocaleTimeString('vi-VN', {
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (error) {
        return 'Vừa xong';
    }
};

export const formatDate = (timestamp) => {
    if (!timestamp) return 'N/A';

    try {
        const date = new Date(timestamp);
        if (isNaN(date.getTime())) return 'N/A';

        return date.toLocaleDateString('vi-VN', {
            day: '2-digit',
            month: '2-digit'
        });
    } catch (error) {
        return 'N/A';
    }
};

export const isValidDate = (timestamp) => {
    if (!timestamp) return false;

    try {
        const date = new Date(timestamp);
        return !isNaN(date.getTime());
    } catch (error) {
        return false;
    }
};

export const getCurrentTimestamp = () => {
    return new Date().toISOString();
};
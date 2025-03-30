// このファイルは不要になりました。削除するか、単純なプレースホルダーにします。

import { Outlet } from 'react-router-dom';
import Layout from './components/Layout';

const App = () => {
  return (
    <Layout>
      <Outlet />
    </Layout>
  );
};

export default App;